import Foundation
import simd

struct NeuralEngine {
    let inputLayer = BitmapInputLayer()
    var outputLayer = DecimalOutputLayer()

    mutating func train(_ sample: MNISTSample) -> [Float] {
        let input = inputLayer.forward(input: sample.image)
        let output = outputLayer.forward(input: input)

        let lossGradient = zip(output, sample.target).map { $0.0 - $0.1 }
        let gradients = outputLayer.computeGradients(input: input, output: output, gradient: lossGradient)

        outputLayer.updateParameters(weightGradients: gradients.weightGradients, biasGradients: gradients.biasGradients, learningRate: 0.1)

        return output
    }

    func evaluate(_ sample: MNISTSample) -> [Float] {
        let input = inputLayer.forward(input: sample.image)
        let output = outputLayer.forward(input: input)
        return output
    }
}
