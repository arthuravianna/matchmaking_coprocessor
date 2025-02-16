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
cd .. && cartesi-coprocessor publish --network devnet
```

```shell
cartesi-coprocessor address-book
```

```shell
cd contracts && cartesi-coprocessor deploy --contract-name CounterCaller --network devnet --constructor-args <task_issuer_address> <machine_hash>
```