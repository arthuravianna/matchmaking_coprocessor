#!/bin/bash

INPUT=$(cast --from-utf8 "`cat pool_of_players.json`")

# private key for 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
# request matchmaking for a pool of players
cast send 0x1429859428C0aBc9C2C47C8Ee9FBaf82cFA0F20f "runExecution(bytes)" ${INPUT} \
    --rpc-url http://localhost:8545 \
    --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80