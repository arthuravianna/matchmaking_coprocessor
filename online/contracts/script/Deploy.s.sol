// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import {Script} from "../lib/forge-std/src/Script.sol";
import {MatchmakingCaller} from "../src/MatchmakingCaller.sol";

contract DeployMatchmakingCaller is Script {
    function run() external returns (MatchmakingCaller) {
        // These values should be replaced with your actual values
        address coprocessorAddress = vm.envAddress("COPROCESSOR_ADDRESS");
        bytes32 machineHash = vm.envBytes32("MACHINE_HASH");

        vm.startBroadcast();
        MatchmakingCaller matchmaking = new MatchmakingCaller(
            coprocessorAddress,
            machineHash
        );
        vm.stopBroadcast();

        return matchmaking;
    }
}