import sys
import os
import numpy as np
from gensim.models import KeyedVectors
from node2vec import Node2Vec
from pandas import read_csv

EMBEDDED_DIMENSIONS = 64

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

    df = read_csv(dataset_csv)
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

