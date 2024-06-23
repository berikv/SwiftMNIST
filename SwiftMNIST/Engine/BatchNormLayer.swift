import Foundation

struct BatchNormLayer {
    var gamma: [Float]
    var beta: [Float]
    var runningMean: [Float]
    var runningVariance: [Float]
    let momentum: Float
    let epsilon: Float

    init(size: Int, momentum: Float = 0.9, epsilon: Float = 1e-5) {
        self.momentum = momentum
        self.epsilon = epsilon

        gamma = [Float](repeating: 1.0, count: size)
        beta = [Float](repeating: 0.0, count: size)
        runningMean = [Float](repeating: 0.0, count: size)
        runningVariance = [Float](repeating: 1.0, count: size)
    }

    func mean(_ input: [Float]) -> Float {
        return input.reduce(0, +) / Float(input.count)
    }

    func variance(_ input: [Float], mean: Float) -> Float {
        return input.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(input.count)
    }

    func forward(input: [Float]) -> [Float] {
        return zip(input, runningMean).enumerated().map { index, tuple in
            let normalized = (tuple.0 - runningMean[index]) / sqrt(runningVariance[index] + epsilon)
            return normalized * gamma[index] + beta[index]
        }
    }

//    mutating func forwardTrain(input: [Float]) -> [Float] {
//        let batchMean = input.mean()
//        let batchVariance = input.variance(mean: batchMean)
//
//        runningMean = runningMean.map { $0 * momentum + batchMean * (1.0 - momentum) }
//        runningVariance = runningVariance.map { $0 * momentum + batchVariance * (1.0 - momentum) }
//
//        return input.enumerated().map { index, input in
//            let normalized = (input - batchMean) / sqrt(batchVariance + epsilon)
//            return normalized * gamma[index] + beta[index]
//        }
//    }

    mutating func forwardTrain(input: [Float]) -> [Float] {
        let batchMean = input.mean()
        let batchVariance = input.variance(mean: batchMean)

        runningMean = zip(runningMean, [Float](repeating: batchMean, count: runningMean.count)).map { $0 * momentum + $1 * (1.0 - momentum) }
        runningVariance = zip(runningVariance, [Float](repeating: batchVariance, count: runningVariance.count)).map { $0 * momentum + $1 * (1.0 - momentum) }

        return input.enumerated().map { index, input in
            let normalized = (input - batchMean) / sqrt(batchVariance + epsilon)
            return normalized * gamma[index] + beta[index]
        }
    }
}
