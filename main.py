import sys
import os
import numpy as np
from gensim.models import KeyedVectors
from node2vec import Node2Vec
import pandas as pd
import networkx as nx
from collections import Counter

EMBEDDED_DIMENSIONS = 64

datafile = './datasets/lol_championship.csv'
data = pd.read_csv(datafile)
data = data.dropna(subset=["champion"])

def create_synergy_graph():
    verticies = []
    win_verticies = []
    lose_verticies = []

    grouped = data.groupby(["gameid", "team"])
    for _, group in grouped:
        champions = group["champion"].tolist()
        for i in range(len(champions)):
            for j in range(i + 1, len(champions)):
                champions_tuple = (champions[i], champions[j])
                if (champions[j], champions[i]) in verticies:
                    champions_tuple = (champions[j], champions[i])
                verticies.append(champions_tuple)

                match_result = group["result"].values[i]
                if match_result == 1:
                    win_verticies.append(champions_tuple)
                elif match_result == 0:
                    lose_verticies.append(champions_tuple)

    win_counter = Counter(win_verticies)
    lose_counter = Counter(lose_verticies)
                
    SynergyGraph = nx.Graph()
    for champion_tuple in verticies:
        edge_weight = win_counter.get(champion_tuple, 0) - lose_counter.get(champion_tuple, 0)
        SynergyGraph.add_edge(champion_tuple[0], champion_tuple[1], weight=edge_weight)

    return SynergyGraph


def create_suppression_graph():
    verticies = []
    defeat_vertices = []
    
    grouped = data.groupby("gameid")
    for _, game in grouped:
        teams = game.groupby("team")
        team_results = {team: group["result"].values[0] for team, group in teams}
        for team, group in teams:
            champions = group["champion"].tolist()
            result = team_results[team]
            for other_team, other_group in teams:
                if team == other_team:
                    continue
                
                other_champions = other_group["champion"].tolist()
                other_result = team_results[other_team]
                
                if result == 1 and other_result == 0: 
                    for champ_win in champions:
                        for champ_lose in other_champions:
                            defeat_vertices.append((champ_win, champ_lose))
                            verticies.append((champ_win, champ_lose))
    
    defeat_counter = Counter(defeat_vertices)
    
    SuppressionGraph = nx.DiGraph()
    for (winner, loser), count in defeat_counter.items():
        SuppressionGraph.add_edge(winner, loser, weight=count)
    
    return SuppressionGraph


def calculate_pick(df, players_dict, heroes_dict):
    pick_rate = np.zeros([n_players, n_heroes])
    pick_win = np.zeros([n_players, n_heroes])

    for player_name in players_dict:
        df_player = df.loc[df['player'] == player_name]
        n_games = len(df_player)
        n_victories = (df_player["result"] == 1).sum()

        for hero_name in heroes_dict:
            df_hero_pick = df_player.loc[df_player['champion'] == hero_name]
            if df_hero_pick.empty: continue

            pick_rate[players_dict[player_name]][heroes_dict[hero_name]] = len(df_hero_pick) / n_games

            hero_victories = (df_hero_pick["result"] == 1).sum()
            pick_win[players_dict[player_name]][heroes_dict[hero_name]] = hero_victories / n_victories
        
    return [pick_rate, pick_win]

def hero2vec(syn_graph, suppr_graph):
    # model with the embeddings of hero synergy graph
    syn_model = gen_or_load_node2vec_model(syn_graph, "hero2vec_syn.txt")
    # model with the embeddings of hero suppression graph
    suppr_model = gen_or_load_node2vec_model(suppr_graph, "hero2vec_suppr.txt")

    syn_graph_node_embeddings = [syn_model[str(node)] for node in syn_graph.nodes()]
    suppr_graph_node_embeddings = [suppr_model[str(node)] for node in suppr_graph.nodes()]

    return [syn_graph_node_embeddings, suppr_graph_node_embeddings]


# gen or load graph's node embeddings
def gen_or_load_node2vec_model(g, model_file):
    model = None

    if os.path.isfile(model_file):
        model = KeyedVectors.load_word2vec_format(model_file, binary=False)
    else:
        node2vec = Node2Vec(g, dimensions=EMBEDDED_DIMENSIONS, walk_length=30, num_walks=200, workers=4, seed=7) # supply seed for reproducibility
        word2vec_model = node2vec.fit(window=10, min_count=1, batch_words=4)
        word2vec_model.wv.save_word2vec_format(model_file)
        model = word2vec_model.wv
    
    return model


def user2vec(n_users, n_heroes, pick_rate, pick_win, syn_graph_node_embeddings, suppr_graph_node_embeddings):
    U_syn = np.empty(n_users)
    U_suppr = np.empty(n_users)

    for user in range(n_users):
        syn_total_sum = np.zeros(EMBEDDED_DIMENSIONS)
        suppr_total_sum = np.zeros(EMBEDDED_DIMENSIONS)
        for hero in range(n_heroes):
            syn_total_sum += pick_rate[user][hero] * syn_graph_node_embeddings[hero]
            suppr_total_sum += pick_win[user][hero] * suppr_graph_node_embeddings[hero]
        
        U_syn[user] = syn_total_sum
        U_suppr[user] = suppr_total_sum
    
    return [U_syn, U_suppr]

            
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Expected one argument: <dataset filename>")
    dataset_csv = sys.argv[1]
    
    if not os.path.isfile(dataset_csv):
        raise Exception("Dataset file does not exists")

    df = pd.read_csv(dataset_csv)
    # remove team rows
    df = df[df["champion"].notna()]

    players = df["player"].unique()
    heroes = df["champion"].unique()

    n_players = len(players)
    n_heroes = len(heroes)
    
    players_dict = dict(zip(players, range(n_players)))
    heroes_dict = dict(zip(heroes, range(n_heroes)))

    [pick_rate, pick_win] = calculate_pick(df, players_dict, heroes_dict)

    # generate graphs
    # TO DO
    # syn_graph_node_embeddings, suppr_graph_node_embeddings = hero2vec(syn_graph, suppr_graph)
    # U_syn, U_suppr = user2vec(n_users, n_heroes, pick_rate, pick_win_rate, syn_graph_node_embeddings, suppr_graph_node_embeddings)

