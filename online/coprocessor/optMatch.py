import json
from MultiHeadAttention import MultiHeadSelfAttention2D
import numpy as np

EMBEDDED_DIMENSIONS = 64
OPTMATCH_FILENAME = "optMatch.json"

class OptMatch:
    def __init__(self):
        self.U_syn = None
        self.U_suppr = None
        self.w_syn = None
        self.w_suppr = None

        # load weights from files
        with open(OPTMATCH_FILENAME) as f:
            content = json.load(f)
            self.U_syn = content["U_syn"]
            self.U_suppr = content["U_suppr"]

            self.w_syn = content["w_syn"]
            self.w_suppr = content["w_suppr"]

    def estimate_match_quality(self, team_a, team_b):
        # o rela SYN
        team_a_syn = np.array([self.U_syn[i] for i in team_a])
        team_b_syn = np.array([self.U_syn[i] for i in team_b])

        multihead_attention = MultiHeadSelfAttention2D(EMBEDDED_DIMENSIONS, 4)
        team_a_T = multihead_attention(team_a_syn)
        team_b_T = multihead_attention(team_b_syn)

        Wt_syn = np.random.rand(1, 64)
        o_rela_syn = np.tanh(Wt_syn@(team_a_T-team_b_T))[0]

        # o rela SUPPR
        team_a_suppr = np.array([self.U_suppr[i] for i in team_a])
        team_b_suppr = np.array([self.U_suppr[i] for i in team_b])

        multihead_attention = MultiHeadSelfAttention2D(EMBEDDED_DIMENSIONS, 4)
        team_a_T = multihead_attention(team_a_suppr)
        team_b_T = multihead_attention(team_b_suppr)

        Wt_suppr = np.random.rand(1, 64)
        o_rela_suppr = np.tanh(Wt_suppr@(team_a_T-team_b_T))[0]

        y = self.w_syn * o_rela_syn + self.w_suppr * o_rela_suppr
        
        return y