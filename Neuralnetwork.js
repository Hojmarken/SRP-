class NeuraltNetværk {
  constructor(lag) {
    this.runningAverage = []
    this.vægte = []
    this.bias = []
    this.læringsHastighed = 0.2
    for (let i = 1; i < lag.length; i++) {
      this.vægte.push(math.random([lag[i], lag[i - 1]], -1, 1))
      this.bias.push(math.random([lag[i], 1], -1, 1))
    }
  }
  fremad(input) {
    this.outputAfLag = []
    let output = math.matrix(input)
    this.outputAfLag.push(output)
    for (let i in this.vægte) {
      output = math.multiply(this.vægte[i], output)
      output = math.add(output, this.bias[i])
      output = output.map(NeuraltNetværk.aktivering)
      this.outputAfLag.push(output)
    }
    return output.valueOf()
  }
  backpropagate(ønsked) {
    let delta = math.subtract(ønsked, this.outputAfLag[this.outputAfLag.length - 1])
    delta = math.dotMultiply(
      delta,
      this.outputAfLag[this.outputAfLag.length - 1].map(x => x * (1 - x))
    )
    this.runningAverage.push(math.sum(math.square(delta)) / 2)
    if (this.runningAverage.length > 100) {
      this.runningAverage.splice(0, 1)
    }
    for (let i = this.outputAfLag.length - 1; i >= 1; i--) {
      let kopiDelta = math.clone(delta)
      if (i > 1) {
        delta = math.multiply(math.transpose(this.vægte[i - 1]), delta)
        delta = math.dotMultiply(delta, this.outputAfLag[i - 1].map(x => x * (1 - x)))
      }
      let deltaVægte = math.multiply(
        math.multiply(kopiDelta, math.transpose(this.outputAfLag[i - 1])),
        this.læringsHastighed
      )
      let deltaBias = math.multiply(kopiDelta, this.læringsHastighed)
      this.vægte[i - 1] = math.add(this.vægte[i - 1], deltaVægte)
      this.bias[i - 1] = math.add(this.bias[i - 1], deltaBias)
    }
  }
  static aktivering(x) {
    return 1 / (1 + Math.pow(Math.E, -x))
  }
}
