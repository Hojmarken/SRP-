let batchLength = 100
let costData = []

function setup() {
  importData()
}

function importData() {
  d3.csv("wheat-seeds.csv").then(function(data) {
    let allData = formatData(data)
    let splitData = divideDataSet(allData)
    let nn = new NeuraltNetværk([7, 3, 3])
    trainNetwork(nn, splitData[0])
    //Åben konsollen for at se klassificeringsrate
    console.log(testNetwork(nn, splitData[1]))
    drawCostData()
  })
}

function drawCostData() {
  var trace1 = {
    x: math.multiply(Array.from(Array(costData.length).keys()), batchLength),
    y: costData,
    type: 'scatter',
  };
  var layout = {
    title: "Læringskurve for neuralt netværk",
    xaxis: {
      title: "Antal træningseksempler"
    },
    yaxis: {
      title: "E"
    }
  }
  Plotly.newPlot('myDiv', [trace1], layout);
}

function formatData(data) {
  let output = []
  let normalisation = []
  for (let i = 0; i < 7; i++) {
    normalisation.push([Infinity, -Infinity, 0])
  }
  for (wheat of data) {
    let values = Object.values(wheat).map(parseFloat)
    for (let i = 0; i < values.length - 1; i++) {
      if (values[i] < normalisation[i][0]) {
        normalisation[i][0] = values[i]
      }
      if (values[i] > normalisation[i][1]) {
        normalisation[i][1] = values[i]
      }
      normalisation[i][2] += values[i] / data.length
    }
  }
  for (wheat of data) {
    let input = []
    let target
    Object.values(wheat).forEach((e, i) => {
      if (i != 7) {
        input.push([(parseFloat(e) - normalisation[i][2]) / max(normalisation[i][1] - normalisation[i][2], normalisation[i][2] - normalisation[i][0])])
      } else {
        target = e == "1" ? [
          [1],
          [0],
          [0]
        ] : e == "2" ? [
          [0],
          [1],
          [0]
        ] : e == "3" ? [
          [0],
          [0],
          [1]
        ] : undefined
      }
    });
    output.push([input, target])
  }
  return output
}

function divideDataSet(data) {
  let dataCopy = data.slice()
  shuffleArray(dataCopy)
  let output = [dataCopy.splice(0, Math.floor(dataCopy.length / 2)), dataCopy]
  return output
}

//Source: https://stackoverflow.com/questions/6274339/how-can-i-shuffle-an-array
function shuffleArray(a) {
  var j, x, i;
  for (i = a.length - 1; i > 0; i--) {
    j = Math.floor(Math.random() * (i + 1));
    x = a[i];
    a[i] = a[j];
    a[j] = x;
  }
  return a;
}


function trainNetwork(nn, dataSet) {
  let epochs = 80000
  for (let i = 0; i < epochs; i++) {
    let data = dataSet[floor(random(dataSet.length))]
    nn.fremad(data[0])
    nn.backpropagate(data[1])
    if (i != 0 && i % batchLength == 0) {
      costData.push(math.mean(nn.runningAverage))
    }
  }
}

function testNetwork(nn, testSet) {
  let total = testSet.length
  let correct = 0
  for (data of testSet) {
    let guess = nn.fremad(data[0])
    let highestIndex = indexOfMax(guess)
    if (data[1][highestIndex][0] == 1) {
      correct += 1
    }
  }
  return correct / total * 100
}

function indexOfMax(vector) {
  let max = -Infinity
  let index = 0
  for (let i = 0; i < vector.length; i++) {
    if (vector[i][0] > max) {
      max = vector[i][0]
      index = i
    }
  }
  return index
}
