import random
import sys
import os
import json
import numpy as np
from gensim.models import KeyedVectors
from node2vec import Node2Vec
import pandas as pd
import networkx as nx

from MultiHeadAttention import MultiHeadSelfAttention2D

EMBEDDED_DIMENSIONS = 64
LEARNING_RATE = 0.1

def create_synergy_graph(df, heroes_dict):
    SynergyGraph = nx.Graph()

    df_groups = df.groupby(["gameid", "team"])
    for _, df_group in df_groups:
        group_heroes = df_group["champion"].to_list()
        result = df_group["result"].to_list()[0]

        for a in range(len(group_heroes)):
            hero_a = heroes_dict[group_heroes[a]]
            for b in range(a+1, len(group_heroes)):
                hero_b = heroes_dict[group_heroes[b]]

                if not SynergyGraph.has_edge(hero_a, hero_b):
                    SynergyGraph.add_edge(hero_a, hero_b, weight=0.0)

                if result == 1:
                    SynergyGraph[hero_a][hero_b]["weight"] += 1
                else:
                    SynergyGraph[hero_a][hero_b]["weight"] -= 1

    # Edge Weights cannot Sum to Zero, otherwise node2vec crashes!!!
    edges = SynergyGraph.edges(data=True)
    for edge in edges:
        if edge[2]["weight"] == 0:
            SynergyGraph[edge[0]][edge[1]]["weight"] = 1e-5 # assign a small value

    return SynergyGraph


def create_suppression_graph(df, heroes_dict):
    SuppressionGraph = nx.DiGraph() # SuppressionGraph[winner][loser] = how many wins

    df_groups = df.groupby(["gameid"])
    for _, df_group in df_groups:
        winning_heroes = df_group.loc[df_group["result"] == 1, "champion"].to_list()
        losing_heroes = df_group.loc[df_group["result"] == 0, "champion"].to_list()

        for hero_a_name in winning_heroes:
            hero_a = heroes_dict[hero_a_name]
            for hero_b_name in losing_heroes:
                hero_b = heroes_dict[hero_b_name]

                if not SuppressionGraph.has_edge(hero_a, hero_b):
                    SuppressionGraph.add_edge(hero_a, hero_b, weight=0)
                SuppressionGraph[hero_a][hero_b]["weight"] += 1

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

            if n_victories > 0:
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
        node2vec = Node2Vec(g, dimensions=EMBEDDED_DIMENSIONS, walk_length=30, num_walks=200, workers=1, seed=7) # supply seed for reproducibility
        word2vec_model = node2vec.fit(window=10, min_count=1, batch_words=4)
        word2vec_model.wv.save_word2vec_format(model_file)
        model = word2vec_model.wv

    return model


def user2vec(n_users, n_heroes, pick_rate, pick_win, syn_graph_node_embeddings, suppr_graph_node_embeddings):
    U_syn = np.empty([n_users, EMBEDDED_DIMENSIONS])
    U_suppr = np.empty([n_users, EMBEDDED_DIMENSIONS])

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
    df = df[df["player"].str.lower() != "team"]

    players = df["player"].unique()
    heroes = df["champion"].unique()

    n_players = len(players)
    n_heroes = len(heroes)

    players_dict = dict(zip(players, range(n_players)))
    heroes_dict = dict(zip(heroes, range(n_heroes)))

    print("Calculating player/hero pick rate and pick win")
    [pick_rate, pick_win] = calculate_pick(df, players_dict, heroes_dict)

    print("Generating Heroes Synergy Graph and Suppression Graph")
    syn_graph = create_synergy_graph(df, heroes_dict)
    suppr_graph = create_suppression_graph(df, heroes_dict)

    print("Generating low-dimension Synergy Graph and Suppression Graph")
    syn_graph_node_embeddings, suppr_graph_node_embeddings = hero2vec(syn_graph, suppr_graph)

    print("Generating Users embedded vectors")
    U_syn, U_suppr = user2vec(n_players, n_heroes, pick_rate, pick_win, syn_graph_node_embeddings, suppr_graph_node_embeddings)

    # print("U_syn:", U_syn)
    # print("U_suppr:", U_suppr)


    Wt_syn = np.random.rand(1, 64)
    Wt_suppr = np.random.rand(1, 64)
    w_syn = random.random()
    w_suppr = random.random()

    multihead_attention_syn = MultiHeadSelfAttention2D(EMBEDDED_DIMENSIONS, 4)
    multihead_attention_suppr = MultiHeadSelfAttention2D(EMBEDDED_DIMENSIONS, 4)

    # Starting the training

    for index in range(int(len(U_syn)/10)):
        first = index*10
        print(first)
        # o rela SYN
        team_a_syn = np.array(U_syn[first:first+5])
        team_b_syn = np.array(U_syn[first+5:first+10])


        team_a_T = multihead_attention_syn.forward(team_a_syn)
        team_b_T = multihead_attention_syn.forward(team_b_syn)


        o_rela_syn = np.tanh(np.dot(Wt_syn, (team_a_T-team_b_T)))[0]

        # o rela SUPPR
        team_a_suppr = np.array(U_suppr[first:first+5])
        team_b_suppr = np.array(U_suppr[first+5:first+10])


        team_a_T = multihead_attention_suppr.forward(team_a_suppr)
        team_b_T = multihead_attention_suppr.forward(team_b_suppr)


        o_rela_suppr = np.tanh(np.dot(Wt_suppr, (team_a_T-team_b_T)))[0]

        #print(f"{w_syn} * {o_rela_syn} + {w_suppr} * {o_rela_suppr}")
        y = w_syn * o_rela_syn + w_suppr * o_rela_suppr
        #print(y)


        # Hidden Unit Error for y
        if (df['result'].iloc[0]):
            d_y = y*(1-y)*(1-y)
        else:
            d_y = y*(1-y)*(-1-y)

        # Hidden Unit Error for o_rela_syn
        d_syn = o_rela_syn * (1 - o_rela_syn) * (w_syn * d_y)

        # Hidden Unit Error for o_rela_suppr
        d_suppr = o_rela_suppr * (1 - o_rela_suppr) * (w_suppr * d_y)

        # Weight Updates l1
        #print(f"w_syn: {w_syn}")
        w_syn = w_syn + LEARNING_RATE * d_y * o_rela_syn
        #print(f"w_syn updated: {w_syn}")
        #print(f"w_suppr: {w_suppr}")
        w_suppr = w_suppr + LEARNING_RATE * d_y * o_rela_suppr
        #print(f"w_suppr updated: {w_suppr}")

        # Weight Updates l2
        #print(f"Wt_syn: {Wt_syn}")
        #print(f"Wt_suppr: {Wt_suppr}")
        for i in range(len(Wt_syn)):
            Wt_syn[i] = Wt_syn[i] + LEARNING_RATE * d_syn * (team_a_T[i] - team_b_T[i])
            Wt_suppr[i] = Wt_suppr[i] + LEARNING_RATE * d_suppr * (team_a_T[i] - team_b_T[i])


        # Hidden Unit Error for (team_a_T[i] - team_b_T[i])

        d_T_syn = np.zeros((1,64))
        d_T_supper = np.zeros((1,64))

        # Hidden Unit Error for team_a_T[i] - team_b_T[i]
        for i in range(len(Wt_syn)):
            # syn backpropagation
            d_T_syn[i] = (team_a_T[i] - team_b_T[i]) * (1 - (team_a_T[i] - team_b_T[i])) * (Wt_syn[i] * d_syn) * LEARNING_RATE
            # suppr backpropagation
            d_T_supper[i] = (team_a_T[i] - team_b_T[i]) * (1 - (team_a_T[i] - team_b_T[i])) * (Wt_suppr[i] * d_suppr) * LEARNING_RATE


        W_plus = np.array([[0.970],[1.214],[1.070],[0.729],[0.873]])
        delta_T_syn = np.dot(W_plus, d_T_syn)
        delta_T_suppr = np.dot(W_plus, d_T_supper)

        # syn backpropagation
        multihead_attention_syn.update_w(np.dot(np.transpose(delta_T_syn), team_a_syn))
        multihead_attention_syn.update_w(np.dot(np.transpose(delta_T_syn), team_b_syn))

        # suppr backpropagation
        multihead_attention_suppr.update_w(np.dot(np.transpose(delta_T_suppr), team_a_suppr))
        multihead_attention_suppr.update_w(np.dot(np.transpose(delta_T_suppr), team_b_suppr))

    WQ_syn, WK_syn, WV_syn = multihead_attention_syn.get_weights()
    WQ_suppr, WK_suppr, WV_suppr = multihead_attention_suppr.get_weights()

    #MISSING: WQ_syn, WK_syn, WV_syn, WQ_suppr, WK_suppr, WV_suppr
    with open("optMatch.json", "w") as opMatchFile:
        print(json.dumps({
                "w_syn": w_syn.item(),
                "w_suppr": w_suppr.item(),
                #"Wt_syn": Wt_syn.to_list(),
                #"Wt_suppr": Wt_suppr.to_list(),
                "U_syn": U_syn.tolist(),
                "U_suppr": U_suppr.tolist()
            }, indent=None),
            file=opMatchFile
        )
