const fs = require("fs");
const path = require("path");
const ff2js = require("ffnet2js");
const GA = require("genetic-algorithm-js");

// import matlab JSON exported feedforwardnet
const data = require("./data/multilayernet.json");

// predict
const predict = (ann) => {
    let outputs = [];
    for (let i = -35; i < 35; i++) {
        // theta in radians
        let theta = (i / 5);

        // predict sin(theta) with ann
        let predictdOut = ann.predict(theta);
        outputs.push(predictdOut.get(0, 0)*10);
    }
    return outputs;
}


const costFunction = (params) => {
    let paramIdx = 0;

    for (let i = 0; i < weights.length; i++) {
        for (let j = 0; j < weights[i].length; j++) {
            for (let k = 0; k < weights[i][j].length; k++) {
                weights[i][j][k] = params[paramIdx] / 255;
                paramIdx++;
            }
        }
        ann.setWeights(weights[i], i + 1);
    }

    for (let i = 0; i < biases.length; i++) {
        for (let j = 0; j < biases[i].length; j++) {
            biases[i][j] = params[paramIdx] / 255;
            paramIdx++;
        }
        ann.setBiases(biases[i], i + 1);
    }

    let gaOut = predict(ann);

    let cost = 0;
    for (let i = 0; i < 70; i++) {
        cost += Math.pow((originalOut[i] - gaOut[i]), 2);
    }
    return -cost;
}
// fs.writeFileSync(path.join(__dirname, "out.json"), JSON.stringify(outputs));

// create an instance of neural net
let ann = new ff2js(data);
const originalOut = predict(ann);

let numParams = 0,
    weights = [],
    biases = [],
    lowLim = [],
    upLim = [];

for (let [idx, layer] of ann.net.layers.entries()) {
    numParams += layer.nNeurons * layer.nInputs + layer.nNeurons;

    weights[idx] = [...new Array(layer.nNeurons)].map(() => new Array(layer.nInputs).fill(0));
    biases[idx] = new Array(layer.nNeurons);
}

for (let [idx, layer] of ann.net.layers.entries()) {
    for (let i = 0; i < layer.weights.val.length; i++) {
        for (let j = 0; j < layer.weights.val[i].length; j++) {
            lowLim.push(Math.floor(layer.weights.val[i][j] * 255) - 20);
            upLim.push(Math.floor(layer.weights.val[i][j] * 255) + 20);
        }
    }
}

for (let [idx, layer] of ann.net.layers.entries()) {
    for (let i = 0; i < layer.biases.val.length; i++) {
        for (let j = 0; j < layer.biases.val[i].length; j++) {
            lowLim.push(Math.floor(layer.biases.val[i][j] * 255) - 20);
            upLim.push(Math.floor(layer.biases.val[i][j] * 255) + 20);
        }
    }
}

let config = {
    numParams,
    lowerLim: lowLim, //new Array(numParams).fill(-1270), // 
    upperLim: upLim, //new Array(numParams).fill(1270), // 
    popSize: 2000,
    maxGen: 100,
    costFunction,
    crossoverRate: 0.6,
    mutationRate: 0.05
}

let ga = new GA(config);
let best = ga.optimize();

console.log(best);

costFunction(best);

fs.writeFileSync(path.join(__dirname, "out.json"), JSON.stringify(predict(ann)));
fs.writeFileSync(path.join(__dirname, "trainedANN.json"), JSON.stringify(ann));
fs.writeFileSync(path.join(__dirname, "original.json"), JSON.stringify(originalOut));