// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "../lib/coprocessor-base-contract/src/CoprocessorAdapter.sol";

contract MatchmakingCaller is CoprocessorAdapter {
    bytes public curr_match;
    event ResultReceived(bytes32 indexed inputPayloadHash, bytes output);

    constructor(address _taskIssuerAddress, bytes32 _machineHash) CoprocessorAdapter(_taskIssuerAddress, _machineHash) {}

    function runExecution(bytes calldata input) external {
        callCoprocessor(input);
    }

    function handleNotice(bytes32 inputPayloadHash, bytes memory notice ) internal override {
        emit ResultReceived(inputPayloadHash, notice);
        curr_match = notice;
    }

    function get() external view returns (bytes memory) {
        return curr_match;
    }
}