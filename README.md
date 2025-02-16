# Matchmaking Coprocessor
Matchmaking in e-sports is the process of pairing players in teams against each other in a fair and competitive manner. Matchmaking systems are crucial in ranked and casual play, ensuring that games are challenging yet enjoyable. So, if done in a centralized manner, the matches can be easily biased, and that's why we propose this project, a matchmaking system using Cartesi Coprocessor. 

We implemented the [OptMatch](https://fuxiailab.github.io/OptMatch/), a generalized, iterative, two-stage, data-driven matchmaking framework that requires minimal product knowledge since it only uses match win/lose/score results. The two stages are offline and online. The offline phase prepares the protocol to predict "good" matches, and the online phase effectively does matchmaking. This means that the coprocessor runs the Online phase, and the offline phase is used to build the coprocessor.

One advantage of the chosen framework is that it is applicable to most gaming products. The framework's precision and generic aspect led us to envision this solution as SaaS (Software as a Service), where a game, decentralized or not, can leverage a decentralized matchmaking system.

## OptMatch
The framework focuses on "arena" games with KVSK players, such as League of Legends and DOTA. It builds relations between the heroes and players to achieve good accuracy. The key advantages of the framework are:
- applicable to most of gaming products, fast and easy to implement
- minimal knowledge about the products and data required
- robust to data drift

### Offline
- extracts two interpersonal relations for representing and understanding tacit coordination interactions among players;
- learns the representation vectors to incorporate the high-order interactions;
- trains a model(i.e., OptMatch-Net) to encode team-up effect and predict the match outcome;

### Online (Coprocessor)
- leverages the representation vectors of players and OptMatch-Net model to maximize the (predicted) gross utilities for the queuing players

## Running
This section details how to execute this project

### Offline
Navigate to the offline directory and create a virtual environment.

```shell
python3 -m venv .venv
. .venv/bin/activate
```

Run the offline/training step
```shell
python3 main.py lol_championship.csv
```

Move the calculated weights to the coprocessor directory. This weights will be used to predict the quality of a match.
```shell
mv optMatch.json ../online/coprocessor
```

### Online (Coprocessor)
```shell
cd coprocessor && cartesi-coprocessor start-devnet
```

```shell
cartesi-coprocessor publish --network devnet
```

```shell
cartesi-coprocessor address-book
```

```shell
cd ../contracts && cartesi-coprocessor deploy --contract-name MatchmakingCaller --network devnet --constructor-args <task_issuer_address> <machine_hash>
```

## INTERACTING
Send a pool of players to generate the matches. Here we are sending a pool of 10 players, i.e. an array of player IDs.
```shell
INPUT=$(cast --from-utf8 "[ 704, 277, 750, 360, 734, 271, 807, 150, 560, 519, 951 ]")
cast send 0x1429859428C0aBc9C2C47C8Ee9FBaf82cFA0F20f "runExecution(bytes)" ${INPUT} \
    --rpc-url http://localhost:8545 \
    --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
```

Wait for the coprossor to finish and then fetch the teams onchain
```shell
cast call --rpc-url http://localhost:8545 0x1429859428C0aBc9C2C47C8Ee9FBaf82cFA0F20f "get()"
```