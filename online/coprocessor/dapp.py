import math
from os import environ
import logging
import requests
import json
import heapq
from itertools import combinations

from optMatch import OptMatch

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

rollup_server = environ["ROLLUP_HTTP_SERVER_URL"]
logger.info(f"HTTP rollup_server url is {rollup_server}")


optMatch = OptMatch()

def str_to_eth_hex(s):
    hex = "0x" + s.encode('utf-8').hex()
    return hex

def emit_notice(data):
    notice_payload = {"payload": data["payload"]}
    response = requests.post(rollup_server + "/notice", json=notice_payload)
    if response.status_code == 200 or response.status_code == 201:
        logger.info(f"Notice emitted successfully with data: {data}")
    else:
        logger.error(f"Failed to emit notice with data: {data}. Status code: {response.status_code}")


def generate_5vs5_matches(player_ids):
    """Generates unique 5v5 matches from a pool of players."""
    for ten_players in combinations(player_ids, 10):
        ten_players = list(ten_players)
        seen_splits = set()
        for team_a in combinations(ten_players, 5):
            team_b = tuple(sorted(set(ten_players) - set(team_a)))
            match = tuple(sorted([tuple(sorted(team_a)), team_b]))
            if match not in seen_splits:
                seen_splits.add(match)
                yield match  # Yield match instead of storing all in memory

def find_best_non_overlapping_matches(player_ids):
    """Finds the best non-overlapping matches based on the quality function."""
    top_k = len(player_ids) / 10
    all_matches = []

    # Step 1: Evaluate all matches and store in a max-heap
    for match in generate_5vs5_matches(player_ids):
        team_a, team_b = match
        logger.info(f"\nTEAM A {team_a}\nTEAM B {team_b}")
        #quality = quality_function(team_a, team_b)
        y = optMatch.estimate_match_quality(team_a, team_b)
        quality = math.cos(y) # closer to 0 the better
        heapq.heappush(all_matches, (-quality, match))  # Max-Heap (negate quality)

    # Step 2: Greedy selection ensuring no player repetition
    selected_matches = []
    used_players = set()

    while all_matches and len(selected_matches) < top_k:
        _, match = heapq.heappop(all_matches)  # Get the highest-quality match
        team_a, team_b = match

        # Check if players are already used
        if not (set(team_a) & used_players or set(team_b) & used_players):
            selected_matches.append(match)
            used_players.update(team_a)
            used_players.update(team_b)

    return selected_matches


def handle_advance(data):
    logger.info(f"Received advance request data {data}")
    payload_hex = data['payload']
    
    try:
        payload_str = bytes.fromhex(payload_hex[2:]).decode('utf-8')
        player_ids = json.loads(payload_str)

        result = find_best_non_overlapping_matches(player_ids)
        emit_notice({ "payload": str_to_eth_hex(json.dumps(result)) })

        return "accept"

    except Exception as error:
        print(f"Error processing payload: {error}")
        return "reject"

handlers = {
    "advance_state": handle_advance,
}

finish = {"status": "accept"}

while True:
    logger.info("Sending finish")
    response = requests.post(rollup_server + "/finish", json=finish)
    logger.info(f"Received finish status {response.status_code}")
    if response.status_code == 202:
        logger.info("No pending rollup request, trying again")
    else:
        rollup_request = response.json()
        data = rollup_request["data"]
        handler = handlers[rollup_request["request_type"]]
        finish["status"] = handler(rollup_request["data"])
