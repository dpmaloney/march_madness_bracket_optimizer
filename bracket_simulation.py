import pandas as pd
import numpy as np
from numpy import int8, float32
from numba import jit, prange
import click


@jit(nopython=True, cache=True)
def pick_one_bracket(public_odds):

    r64_matchups = public_odds[0].reshape(32, 2)
    r32_raw = public_odds[1]

    r32 = np.zeros(32)
    r32_teams = np.zeros(32, dtype=int8)

    r64_randoms = np.random.random(32)
    r64_odds = r64_matchups[:, 0] / r64_matchups.sum(axis=1)
    r64_winners = r64_randoms > r64_odds
    r64_winners = r64_winners.astype(int8)

    for i in range(32):
        r32_teams[i] = i * 2 + r64_winners[i]
        r32[i] = r32_raw[i * 2 + r64_winners[i]]

    r32_matchups = r32.reshape(16, 2)

    r16_raw = public_odds[2]
    r16 = np.zeros(16)
    r16_teams = np.zeros(16, dtype=int8)
    r32_randoms = np.random.random(16)
    r32_odds = r32_matchups[:, 0] / r32_matchups.sum(axis=1)
    r32_winners = r32_randoms > r32_odds
    r32_winners = r32_winners.astype(int8)

    for i in range(16):
        r16_teams[i] = r32_teams[i * 2 + r32_winners[i]]
        r16[i] = r16_raw[r32_teams[i * 2 + r32_winners[i]]]

    r16_matchups = r16.reshape(8, 2)
    r8_raw = public_odds[3]
    r8 = np.zeros(8)
    r8_teams = np.zeros(8, dtype=int8)
    r16_randoms = np.random.random(8)
    r16_odds = r16_matchups[:, 0] / r16_matchups.sum(axis=1)
    r16_winners = r16_randoms > r16_odds
    r16_winners = r16_winners.astype(int8)

    for i in range(8):
        r8_teams[i] = r16_teams[i * 2 + r16_winners[i]]
        r8[i] = r8_raw[r16_teams[i * 2 + r16_winners[i]]]

    r8_matchups = r8.reshape(4, 2)
    r4_raw = public_odds[4]
    r4 = np.zeros(4)
    r4_teams = np.zeros(4, dtype=int8)
    r8_randoms = np.random.random(4)
    r8_odds = r8_matchups[:, 0] / r8_matchups.sum(axis=1)
    r8_winners = r8_randoms > r8_odds
    r8_winners = r8_winners.astype(int8)

    for i in range(4):
        r4_teams[i] = r8_teams[i * 2 + r8_winners[i]]
        r4[i] = r4_raw[r8_teams[i * 2 + r8_winners[i]]]

    r4_matchups = r4.reshape(2, 2)
    r2_raw = public_odds[5]
    r2 = np.zeros(2)
    r2_teams = np.zeros(2, dtype=int8)
    r4_randoms = np.random.random(2)
    r4_odds = r4_matchups[:, 0] / r4_matchups.sum(axis=1)
    r4_winners = r4_randoms > r4_odds
    r4_winners = r4_winners.astype(int8)

    for i in range(2):
        r2_teams[i] = r4_teams[i * 2 + r4_winners[i]]
        r2[i] = r2_raw[r4_teams[i * 2 + r4_winners[i]]]

    r2_matchups = r2.reshape(1, 2)
    r2_randoms = np.random.random(1)
    r2_odds = r2_matchups[:, 0] / r2_matchups.sum(axis=1)
    r2_winners = r2_randoms > r2_odds
    r2_winners = r2_winners.astype(int8)

    final_winner = r2_teams[r2_winners[0]]
    fw = np.zeros(1)
    fw[0] = final_winner

    rw = np.zeros(63, dtype=int8)
    rw[:32] = r32_teams
    rw[32:48] = r16_teams
    rw[48:56] = r8_teams
    rw[56:60] = r4_teams
    rw[60:62] = r2_teams
    rw[62] = final_winner
    return rw


@jit(nopython=True, cache=True)
def simulate_tournament_results(teams_odds):

    r64_matchups = teams_odds[0].reshape(32, 2)
    r32_raw = teams_odds[1]

    r32 = np.zeros(32)
    r32_teams = np.zeros(32, dtype=int8)

    r64_randoms = np.random.random(32)
    r64_odds = r64_matchups[:, 0] / r64_matchups.sum(axis=1)
    r64_winners = r64_randoms > r64_odds
    r64_winners = r64_winners.astype(int8)

    for i in range(32):
        r32_teams[i] = i * 2 + r64_winners[i]
        r32[i] = r32_raw[i * 2 + r64_winners[i]]

    r32_matchups = r32.reshape(16, 2)

    r16_raw = teams_odds[2]
    r16 = np.zeros(16)
    r16_teams = np.zeros(16, dtype=int8)
    r32_randoms = np.random.random(16)
    r32_odds = r32_matchups[:, 0] / r32_matchups.sum(axis=1)
    r32_winners = r32_randoms > r32_odds
    r32_winners = r32_winners.astype(int8)

    for i in range(16):
        r16_teams[i] = r32_teams[i * 2 + r32_winners[i]]
        r16[i] = r16_raw[r32_teams[i * 2 + r32_winners[i]]]

    r16_matchups = r16.reshape(8, 2)
    r8_raw = teams_odds[3]
    r8 = np.zeros(8)
    r8_teams = np.zeros(8, dtype=int8)
    r16_randoms = np.random.random(8)
    r16_odds = r16_matchups[:, 0] / r16_matchups.sum(axis=1)
    r16_winners = r16_randoms > r16_odds
    r16_winners = r16_winners.astype(int8)

    for i in range(8):
        r8_teams[i] = r16_teams[i * 2 + r16_winners[i]]
        r8[i] = r8_raw[r16_teams[i * 2 + r16_winners[i]]]

    r8_matchups = r8.reshape(4, 2)
    r4_raw = teams_odds[4]
    r4 = np.zeros(4)
    r4_teams = np.zeros(4, dtype=int8)
    r8_randoms = np.random.random(4)
    r8_odds = r8_matchups[:, 0] / r8_matchups.sum(axis=1)
    r8_winners = r8_randoms > r8_odds
    r8_winners = r8_winners.astype(int8)

    for i in range(4):
        r4_teams[i] = r8_teams[i * 2 + r8_winners[i]]
        r4[i] = r4_raw[r8_teams[i * 2 + r8_winners[i]]]

    r4_matchups = r4.reshape(2, 2)
    r2_raw = teams_odds[5]
    r2 = np.zeros(2)
    r2_teams = np.zeros(2, dtype=int8)
    r4_randoms = np.random.random(2)
    r4_odds = r4_matchups[:, 0] / r4_matchups.sum(axis=1)
    r4_winners = r4_randoms > r4_odds
    r4_winners = r4_winners.astype(int8)

    for i in range(2):
        r2_teams[i] = r4_teams[i * 2 + r4_winners[i]]
        r2[i] = r2_raw[r4_teams[i * 2 + r4_winners[i]]]

    r2_matchups = r2.reshape(1, 2)
    r2_randoms = np.random.random(1)
    r2_odds = r2_matchups[:, 0] / r2_matchups.sum(axis=1)
    r2_winners = r2_randoms > r2_odds
    r2_winners = r2_winners.astype(int8)

    final_winner = r2_teams[r2_winners[0]]

    rw = np.zeros(63, dtype=int8)
    rw[:32] = r32_teams
    rw[32:48] = r16_teams
    rw[48:56] = r8_teams
    rw[56:60] = r4_teams
    rw[60:62] = r2_teams
    rw[62] = final_winner

    return rw


@jit(nopython=True, cache=True)
def score_bracket(bracket, tournament_result):
    score = 0

    for i in range(63):
        if bracket[i] == tournament_result[i]:
            if i < 32:
                score += 1
            elif i < 48:
                score += 2
            elif i < 56:
                score += 4
            elif i < 60:
                score += 8
            elif i < 62:
                score += 16
            else:
                score += 32
        # print(score)

    return score


@jit(nopython=True, parallel=True, cache=True)
def simulate_pool(teams_odds, public_odds, pool_size, payouts, num_tournaments=1000):
    pool = np.zeros((pool_size, 63), dtype=int8)
    for i in prange(pool_size):
        pool[i] = pick_one_bracket(public_odds)

    # winnings_tally = np.zeros(pool_size, dtype=float32)
    all_winnings = np.zeros((num_tournaments, pool_size), dtype=float32)
    for i in prange(num_tournaments):
        tournament_result = simulate_tournament_results(teams_odds)
        scores_list = np.zeros(pool_size, dtype=float32)
        for z in prange(pool_size):
            bracket_score = score_bracket(pool[z], tournament_result)
            scores_list[z] = bracket_score

        unique_scores = np.unique(scores_list)
        unique_scores.sort()
        unique_scores = unique_scores[::-1]

        for p in range(len(payouts)):
            payout = payouts[p]
            count = (scores_list == unique_scores[p]).sum()
            is_max = np.argwhere(scores_list == unique_scores[p])
            for j in range(count):
                all_winnings[i][is_max[j]] = payout / count

    summed_winnings = np.sum(all_winnings, axis=0)
    average_winnings = summed_winnings / num_tournaments

    order = np.argsort(average_winnings)[::-1]

    # average_winnings = [(pool[i], average_winnings[i]) for i in range(pool_size)]
    # average_winnings = sorted(average_winnings, key=lambda x: x[1], reverse=True)

    best_winnings = average_winnings[order[0]]
    best_pool = pool[order[0]]

    out = np.zeros(64, dtype=float32)
    out[:63] = best_pool
    out[63] = best_winnings
    return out


@jit(nopython=True, parallel=True, cache=True)
def get_best_brackets(
    teams_odds, public_odds, pool_size, payouts, num_tournaments=1000, num_pool_sims=10
):
    
    best_brackets = np.zeros((num_pool_sims, 63), dtype=int8)
    best_scores = np.zeros(num_pool_sims, dtype=float32)
    for i in prange(num_pool_sims):
        outcome = simulate_pool(
            teams_odds, public_odds, pool_size, payouts, num_tournaments
        )
        bb = outcome[0:63].astype(int8)
        bs = outcome[63]
        best_brackets[i] = bb
        best_scores[i] = bs

    combined = [(brack, score) for brack, score in zip(best_brackets, best_scores)]

    return combined


@click.command()
@click.option("--pool_size", default=10, help="Number of people in your pool")
@click.option(
    "--payouts",
    prompt="The payouts for the winning places",
    help="For $1000 to first, and $500 to second, enter 1000,500",
)
@click.option(
    "--num_tournaments",
    default=1000,
    help="Number of tournament outcomes to simulate per pool",
)
@click.option("--num_pool_sims", default=10, help="Number of pools to simulate")
def find_best_brackets(pool_size, payouts, num_tournaments, num_pool_sims):
    payout_list = payouts.split(",")
    payouts = [int(x) for x in payout_list]

    game_order = pd.read_csv("data/tournament_layout.csv")
    order = game_order["team_name"].tolist()

    total_data = pd.read_csv("data/final_data.csv")

    teams_odds = np.zeros((6, 64))
    public_odds = np.zeros((6, 64))
    teamid_to_name = {}
    for i in range(len(order)):
        name = order[i]
        team_id = i
        teamid_to_name[team_id] = name
        data = total_data[total_data["Team Name"] == name]
        
        for j in range(6):
            z = 5 - j
            
            teams_odds[j][i] = data["Odds"].values[z]
            public_odds[j][i] = data["Public Pick%"].values[z]
            
    print(f"Simulating {num_pool_sims} pools...")
    best_brackets = get_best_brackets(
        teams_odds, public_odds, pool_size, payouts, num_tournaments, num_pool_sims
    )

    best_brackets = sorted(best_brackets, key=lambda x: x[1], reverse=True)
    with open("best_brackets.txt", "w+") as f:
        for i in range(len(best_brackets)):
            r32_winners = [teamid_to_name.get(x) for x in best_brackets[i][0][0:32]]
            r16_winners = [teamid_to_name.get(x) for x in best_brackets[i][0][32:48]]
            r8_winners = [teamid_to_name.get(x) for x in best_brackets[i][0][48:56]]
            r4_winners = [teamid_to_name.get(x) for x in best_brackets[i][0][56:60]]
            r2_winners = [teamid_to_name.get(x) for x in best_brackets[i][0][60:62]]
            final_winner = [teamid_to_name.get(best_brackets[i][0][62])]

            f.write("Bracket " + str(i + 1) + "\n")
            f.write("R32 Winners: " + str(r32_winners) + "\n")
            f.write("R16 Winners: " + str(r16_winners) + "\n")
            f.write("R8 Winners: " + str(r8_winners) + "\n")
            f.write("R4 Winners: " + str(r4_winners) + "\n")
            f.write("R2 Winners: " + str(r2_winners) + "\n")
            f.write("Final Winner: " + str(final_winner) + "\n")
            f.write("Expected Winnings: " + str(best_brackets[i][1]) + "\n")
            f.write("\n")


if __name__ == "__main__":
    find_best_brackets()
