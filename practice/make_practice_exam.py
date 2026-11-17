"""Build the practice exam: Metrica's 3 public sample matches, rewritten in the competition's
tracking format and made about as hard as the competition match.

    ../.venv/bin/python make_practice_exam.py

For each sample match it writes exam/sample_game_<n>/:

    sample_game_<n>_atd.xml   tracking metadata (FIFA/EPTS), the same layout as the competition's
    sample_game_<n>_atd.txt   tracking, one line per frame, the same layout as the competition's:
                              "<frame_id>:<x,y,speed>;... 46 track slots ...:<ball x,y,speed>"
    info.json                 frame count, graded halves, and what was done to the match

The answers stay in answers/sample_game_<n>.csv (built by build_answer_key.py).

Only facts the organisers published about the competition match are used (DATA.md), never the
match data itself:

  - 46 anonymous track slots named "Track 1".."Track 46", 23 per team, with the two teams mixed
    across the slot numbers. A player's track is cut into pieces and each piece goes into a random
    free slot of his team, so a slot is not a person.
  - Players are briefly lost now and then, so about 20.5 of 22 are tracked on average.
  - The ball is lost DURING PLAY far more often than in the samples. The samples already miss the
    ball on 32-41% of graded frames, close to the competition's 41%, but almost only while play is
    stopped, so "no ball = stopped" works far better on them than it can on the competition match,
    whose ball is still reported on about half of the frames with no play at all (DATA.md: 56.9%
    seen on all frames, 59.3% on graded ones). If stoppages there lose the ball about as often as
    those no-play frames do, about a third of the in-play frames have no ball. So gaps are added
    -- more often where the ball is in a crowd or moving fast, as DATA.md describes -- until 35% of
    in-play frames miss the ball. This 35% is our estimate, not a published number.
  - Speed in m/s, NaN when missing. Positions normalised 0-1.

Everything random uses a fixed seed, so running this again writes exactly the same files.
"""

import csv
import json
import os

import numpy as np

from check_answer_key import load_tracking

HERE = os.path.dirname(os.path.abspath(__file__))
EXAM = os.path.join(HERE, "exam")
ANSWERS = os.path.join(HERE, "answers")
FPS = 25
PITCH = np.array([105.0, 68.0])
GAMES = (1, 2, 3)

# Facts published in DATA.md about the competition match.
N_SLOTS = 46

# Our choices, so the samples behave like those facts. Not measured on the competition match.
BALL_MISSING_IN_PLAY = 0.35   # estimate from DATA.md's visibility numbers, see the top of the file
LOSS_EVERY_S = 40.0     # a player is lost about once every 40 s...
LOSS_MEAN_S = 3.0       # ...for about 3 s, so ~93% (about 20.5 of 22) are tracked
SWAP_MEAN_S = 20.0      # a track piece lasts about 20 s before the player lands in a new slot
GAP_MEDIAN_S = 1.0      # added ball gaps: typically 1 s, some much longer...
GAP_SPREAD = 1.0        # ...(log-normal spread)...
GAP_MAX_S = 15.0        # ...but never more than 15 s
CROWD_M = 3.0           # "in a crowd": 2 or more players within 3 m of the ball
FAST_MS = 15.0          # "moving fast": ball faster than 15 m/s (long pass, shot, in the air)
SEED = 2027


# ---------------------------------------------------------------------------------------------
# Reading a sample match
# ---------------------------------------------------------------------------------------------

def read_answers(game):
    with open(os.path.join(ANSWERS, "sample_game_%d.csv" % game)) as handle:
        return np.array([row["label"] for row in csv.DictReader(handle)])


def load_arrays(game, n_frames):
    """(ball[n,2], players[n,p,2], is_home[p], period[n]) in normalised 0-1 coordinates.

    Row i is tracking frame i + 1, like the answer file. NaN where something is not seen.
    """
    dataset = load_tracking(game)
    df = dataset.to_df()
    home = next(t for t in dataset.metadata.teams if str(t.ground).lower().endswith("home"))
    ids, is_home = [], []
    for team in dataset.metadata.teams:
        for player in team.players:
            if "%s_x" % player.player_id in df.columns:
                ids.append(str(player.player_id))
                is_home.append(team is home)

    rows = df["frame_id"].to_numpy(int) - 1
    if rows.min() < 0 or rows.max() >= n_frames:
        raise ValueError("sample game %d: tracking frames do not fit the answer file" % game)

    ball = np.full((n_frames, 2), np.nan)
    ball[rows] = df[["ball_x", "ball_y"]].to_numpy(float)
    players = np.full((n_frames, len(ids), 2), np.nan)
    for k, pid in enumerate(ids):
        players[rows, k] = df[[pid + "_x", pid + "_y"]].to_numpy(float)
    period = np.zeros(n_frames, dtype=int)
    period[rows] = df["period_id"].fillna(0).to_numpy(int)
    return ball, players, np.array(is_home), period


# ---------------------------------------------------------------------------------------------
# Making it harder
# ---------------------------------------------------------------------------------------------

def speed(track):
    """m/s over +/- 2 frames, NaN wherever a position is missing."""
    metres = track * PITCH
    out = np.full(len(track), np.nan)
    out[2:-2] = np.linalg.norm(metres[4:] - metres[:-4], axis=1) * FPS / 4.0
    return out


def runs(mask):
    """[(start, stop)] of the True stretches of a boolean array, stop exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def lost_moments(n_frames, rng):
    """Moments a player is briefly lost by the tracker."""
    lost = np.zeros(n_frames, dtype=bool)
    t = 0.0
    while True:
        t += rng.exponential(LOSS_EVERY_S * FPS)
        if t >= n_frames:
            return lost
        length = max(1, int(rng.exponential(LOSS_MEAN_S * FPS)))
        lost[int(t):int(t) + length] = True
        t += length


def track_pieces(visible, rng):
    """Cut a player's visible stretches into pieces of about SWAP_MEAN_S seconds."""
    pieces = []
    for start, stop in runs(visible):
        t = start
        while t < stop:
            end = min(stop, t + max(1, int(rng.exponential(SWAP_MEAN_S * FPS))))
            pieces.append((t, end))
            t = end
    return pieces


def to_slots(players, is_home, rng):
    """Spread the players over 46 anonymous slots, the way a tracker loses and re-finds them.

    Returns (slot_xy[n,46,2], slot_speed[n,46], slot_is_home[46], n_pieces).
    """
    n_frames = len(players)
    slot_is_home = rng.permutation(np.array([True] * (N_SLOTS // 2)
                                            + [False] * (N_SLOTS - N_SLOTS // 2)))
    slot_xy = np.full((n_frames, N_SLOTS, 2), np.nan)
    slot_speed = np.full((n_frames, N_SLOTS), np.nan)

    pieces = []
    for k in range(players.shape[1]):
        visible = ~np.isnan(players[:, k, 0]) & ~lost_moments(n_frames, rng)
        pieces.extend((start, stop, k) for start, stop in track_pieces(visible, rng))
    pieces.sort()

    free_from = np.zeros(N_SLOTS, dtype=int)
    for start, stop, k in pieces:
        options = np.flatnonzero((slot_is_home == is_home[k]) & (free_from <= start))
        if not len(options):
            raise RuntimeError("ran out of track slots")
        slot = rng.choice(options)
        free_from[slot] = stop
        slot_xy[start:stop, slot] = players[start:stop, k]
        slot_speed[start:stop, slot] = speed(players[:, k])[start:stop]
    return slot_xy, slot_speed, slot_is_home, len(pieces)


def hide_ball(ball, players, period, in_play, rng):
    """Copy of the ball track with extra gaps until BALL_MISSING_IN_PLAY of in-play frames miss it.

    Gaps start more often where the ball is in a crowd or moving fast. The answers (which frames
    are in play) are only used to count the share, never to place the gaps.
    """
    n_frames = len(ball)
    hidden = np.isnan(ball[:, 0])
    with np.errstate(invalid="ignore"):
        near = np.linalg.norm((players - ball[:, None, :]) * PITCH, axis=2) < CROWD_M
        crowd = near.sum(axis=1) >= 2
        fast = speed(ball) > FAST_MS
    weight = (1.0 + 2.0 * crowd + 2.0 * fast) * (~hidden & (period > 0))

    target = int(np.ceil(BALL_MISSING_IN_PLAY * in_play.sum()))
    missing = int((hidden & in_play).sum())
    while missing < target:
        for start in rng.choice(n_frames, size=1000, p=weight / weight.sum()):
            length = int(np.clip(rng.lognormal(np.log(GAP_MEDIAN_S * FPS), GAP_SPREAD),
                                 3, GAP_MAX_S * FPS))
            stretch = slice(start, min(n_frames, start + length))
            missing += int((~hidden[stretch] & in_play[stretch]).sum())
            hidden[stretch] = True
            if missing >= target:
                break

    out = ball.copy()
    out[hidden] = np.nan
    return out


# ---------------------------------------------------------------------------------------------
# Writing the competition format
# ---------------------------------------------------------------------------------------------

def groups(xy, spd):
    """One 'x,y,speed' text per frame, 'NaN,NaN,NaN' where not seen."""
    out = np.full(len(xy), "NaN,NaN,NaN", dtype=object)
    seen = np.flatnonzero(~np.isnan(xy[:, 0]))
    out[seen] = ["%.5f,%.5f,%s" % (x, y, "NaN" if s != s else "%.2f" % s)
                 for x, y, s in zip(xy[seen, 0], xy[seen, 1], spd[seen])]
    return out


def write_raw(path, slot_xy, slot_speed, ball):
    columns = [groups(slot_xy[:, s], slot_speed[:, s]) for s in range(N_SLOTS)]
    ball_text = groups(ball, speed(ball))
    with open(path, "w", newline="\n") as handle:
        for i, slots in enumerate(zip(*columns)):
            handle.write("%d:%s:%s\n" % (i + 1, ";".join(slots), ball_text[i]))


def parameter(name, value):
    return ("<ProviderParameter><Name>%s</Name><Value>%s</Value></ProviderParameter>"
            % (name, value))


def write_metadata(path, game, n_frames, period, slot_is_home, home_attacks_right):
    first_half = np.flatnonzero(period == 1) + 1
    second_half = np.flatnonzero(period == 2) + 1
    home_dir, away_dir = (("left_to_right", "right_to_left") if home_attacks_right
                          else ("right_to_left", "left_to_right"))

    players = "\n".join(
        '      <Player id="%d" teamId="%s"><Name>Track %d</Name><ShirtNumber>%d</ShirtNumber>'
        "</Player>" % (s + 1, "HOME" if slot_is_home[s] else "AWAY", s + 1, s + 1)
        for s in range(N_SLOTS))
    channels = "\n".join(
        '      <PlayerChannel channelId="%s" id="player%d_%s" playerId="%d"/>' % (c, s + 1, c, s + 1)
        for s in range(N_SLOTS) for c in ("x", "y", "s"))
    slot_format = "\n".join(
        '        <SplitRegister separator=",">'
        '<PlayerChannelRef playerChannelId="player{0}_x"/>'
        '<PlayerChannelRef playerChannelId="player{0}_y"/>'
        '<PlayerChannelRef playerChannelId="player{0}_s"/></SplitRegister>'.format(s + 1)
        for s in range(N_SLOTS))

    xml = """<?xml version="1.0" encoding="utf-8"?>
<main>
  <Metadata>
    <GlobalConfig>
      <FileName>sample_game_{game}_atd</FileName>
      <TrackingType>Optical</TrackingType>
      <ProviderName>Metrica Sports</ProviderName>
      <FrameRate>{fps}</FrameRate>
      <ProviderGlobalParameters>
        {p1}
        {p2}
        {p3}
        {p4}
      </ProviderGlobalParameters>
    </GlobalConfig>
    <Sessions>
      <Session id="session1">
        <SessionType>Match</SessionType>
        <MatchParameters>
          <Score idLocalTeam="HOME" idVisitingTeam="AWAY">
            <LocalTeamScore>0</LocalTeamScore>
            <VisitingTeamScore>0</VisitingTeamScore>
          </Score>
        </MatchParameters>
        <FieldSize>
          <Width>105</Width>
          <Height>68</Height>
        </FieldSize>
      </Session>
    </Sessions>
    <Teams>
      <Team id="HOME"><Name>Sample {game} Home</Name><ProviderTeamsParameters>{home_dir}</ProviderTeamsParameters></Team>
      <Team id="AWAY"><Name>Sample {game} Away</Name><ProviderTeamsParameters>{away_dir}</ProviderTeamsParameters></Team>
    </Teams>
    <Players>
{players}
    </Players>
    <Devices>
      <Device id="device1">
        <Name>Device 1</Name>
        <Sensors>
          <Sensor id="position">
            <Name>Position</Name>
            <Channels>
              <Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>
              <Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel>
            </Channels>
          </Sensor>
          <Sensor id="speed">
            <Name>Speed</Name>
            <Channels>
              <Channel id="s"><Name>speed</Name><Unit>meters/second</Unit></Channel>
            </Channels>
          </Sensor>
        </Sensors>
      </Device>
    </Devices>
    <PlayerChannels>
{channels}
    </PlayerChannels>
  </Metadata>
  <DataFormatSpecifications>
    <DataFormatSpecification endFrame="{n}" separator=":" startFrame="1">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
{slot_format}
      </SplitRegister>
      <SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/><BallChannelRef channelId="s"/></SplitRegister>
    </DataFormatSpecification>
  </DataFormatSpecifications>
</main>
""".format(game=game, fps=FPS, n=n_frames, players=players, channels=channels,
           slot_format=slot_format,
           p1=parameter("first_half_start", first_half[0]),
           p2=parameter("first_half_end", first_half[-1]),
           p3=parameter("second_half_start", second_half[0]),
           p4=parameter("second_half_end", second_half[-1]),
           home_dir=parameter("attack_direction_first_half", home_dir),
           away_dir=parameter("attack_direction_first_half", away_dir))
    with open(path, "w") as handle:
        handle.write(xml)


# ---------------------------------------------------------------------------------------------

def mean_tracked(xy, graded):
    return float((~np.isnan(xy[graded, :, 0])).sum(axis=1).mean())


def build(game):
    rng = np.random.RandomState(SEED + game)
    answers = read_answers(game)
    n_frames = len(answers)
    graded = answers != "X"
    in_play = (answers == "H") | (answers == "A")
    stopped = answers == "D"

    ball, players, is_home, period = load_arrays(game, n_frames)
    slot_xy, slot_speed, slot_is_home, n_pieces = to_slots(players, is_home, rng)
    exam_ball = hide_ball(ball, players, period, in_play, rng)

    def missing(where):
        return {"original": round(float(np.isnan(ball[where, 0]).mean()), 3),
                "exam": round(float(np.isnan(exam_ball[where, 0]).mean()), 3)}

    first = np.flatnonzero(period == 1)[:FPS * 10]
    home_x = np.nanmean(players[first][:, is_home, 0])
    home_attacks_right = home_x < 0.5

    folder = os.path.join(EXAM, "sample_game_%d" % game)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    write_metadata(os.path.join(folder, "sample_game_%d_atd.xml" % game), game, n_frames,
                   period, slot_is_home, home_attacks_right)
    write_raw(os.path.join(folder, "sample_game_%d_atd.txt" % game), slot_xy, slot_speed,
              exam_ball)

    info = {
        "source": "Metrica Sports public sample data, Sample_Game_%d "
                  "(github.com/metrica-sports/sample-data)" % game,
        "n_frames": n_frames,
        "halves": [[start, stop - 1] for start, stop in runs(graded)],
        "players_tracked_graded": {"original": round(mean_tracked(players, graded), 1),
                                   "exam": round(mean_tracked(slot_xy, graded), 1)},
        "ball_missing_graded": missing(graded),
        "ball_missing_in_play": missing(in_play),
        "ball_missing_stopped": missing(stopped),
        "track_pieces": n_pieces,
        "seed": SEED + game,
    }
    with open(os.path.join(folder, "info.json"), "w") as handle:
        json.dump(info, handle, indent=2)
    return info


def main():
    for game in GAMES:
        info = build(game)
        print("Sample game %d -> exam/sample_game_%d/" % (game, game))
        print("    frames %d, graded halves %s" % (info["n_frames"], info["halves"]))
        print("    players tracked (graded frames)  %.1f -> %.1f of 22"
              % (info["players_tracked_graded"]["original"], info["players_tracked_graded"]["exam"]))
        for where in ("in_play", "stopped", "graded"):
            share = info["ball_missing_" + where]
            print("    ball missing (%-8s frames)    %3.0f%% -> %3.0f%%"
                  % (where.replace("_", " "), 100 * share["original"], 100 * share["exam"]))
        print("    a player's track is cut into %d pieces spread over %d slots"
              % (info["track_pieces"], N_SLOTS))
        print()


if __name__ == "__main__":
    main()
