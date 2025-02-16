# Matchmaking Coprocessor
...

## OptMatch

### Offline
### Online (Coprocessor)

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
```shell
INPUT=$(cast --from-utf8 "[ 704, 277, 750, 360, 734, 271, 807, 150, 560, 519, 951 ]")
cast send 0x1429859428C0aBc9C2C47C8Ee9FBaf82cFA0F20f "runExecution(bytes)" ${INPUT} \
    --rpc-url http://localhost:8545 \
    --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
```