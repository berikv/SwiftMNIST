import Foundation
import simd

struct NeuralEngine {

    let learningRateStart: Float = 0.1
    let learningRateFactor: Float = 0.6

    private(set) var learningRate: Float = 0.1

    var epoch: Int = 0 {
        didSet {
            learningRate = learningRateStart * pow(learningRateFactor, Float(epoch))
        }
    }

    private let inputLayer = BitmapInputLayer()
    private var hiddenLayer = HiddenLayer()
    private var outputLayer = DecimalOutputLayer()


    mutating func train(_ sample: MNISTSample) -> [Float] {

//        Single layer approach:
//        let input = inputLayer.forward(input: sample.image)
//        let output = outputLayer.forward(input: input)
//
//        let lossGradient = zip(output, sample.target).map { $0.0 - $0.1 }
//        let gradients = outputLayer.computeGradients(input: input, gradient: lossGradient)
//
//        outputLayer.updateParameters(weightGradients: gradients.weightGradients, biasGradients: gradients.biasGradients, learningRate: 0.1)
//
//        return output

        // Forward pass
        let inputLayerOutput = inputLayer.forward(input: sample.image)
        let hiddenLayerOutput = hiddenLayer.forward(input: inputLayerOutput)
        try! trapOutOfBounds(hiddenLayerOutput, bounds: -50...50)
        let outputLayerOutput = outputLayer.forward(input: hiddenLayerOutput)

        // Compute the gradient of the loss with respect to the output
        //        Why This Works:
        //
        //            •    Softmax: It converts logits to probabilities. The softmax function ensures that the predicted probabilities sum to 1.
        //            •    Cross-Entropy Loss: It measures the difference between the predicted probabilities and the actual labels. By taking the gradient of this loss with respect to the logits, we get a vector that tells us how to adjust the logits to reduce the loss.
        let lossGradient = zip(outputLayerOutput, sample.target).map { $0.0 - $0.1 }
        try! trapOutOfBounds(lossGradient)

        // Backward pass through the output layer to get the gradient with respect to hidden layer output
        let hiddenLayerGradient = outputLayer.backward(input: hiddenLayerOutput, output: outputLayerOutput, gradient: lossGradient)
        try! trapOutOfBounds(hiddenLayerGradient, bounds: -50...50)

        // Compute gradients and update parameters for the output layer
        let outputGradients = outputLayer.computeGradients(input: hiddenLayerOutput, gradient: lossGradient)
        try! trapOutOfBounds(outputGradients.biasGradients)
        try! trapOutOfBounds(outputGradients.weightGradients, bounds: -50...50)
        outputLayer.updateParameters(weightGradients: outputGradients.weightGradients, biasGradients: outputGradients.biasGradients, learningRate: learningRate)

        // Compute gradients and update parameters for the hidden layer
        let hiddenLayerGradients = hiddenLayer.computeGradients(input: inputLayerOutput, gradient: hiddenLayerGradient)
        try! trapOutOfBounds(hiddenLayerGradients.biasGradients, bounds: -50...50)
        try! trapOutOfBounds(hiddenLayerGradients.weightGradients, bounds: -100...100)
        hiddenLayer.updateParameters(weightGradients: hiddenLayerGradients.weightGradients, biasGradients: hiddenLayerGradients.biasGradients, learningRate: learningRate)

        // Backward pass through the hidden layer is not needed as the input layer does not learn
//        let inputLayerGradient = hiddenLayer.backward(input: inputLayerOutput, output: hiddenLayerOutput, gradient: hiddenLayerGradient)

        return outputLayerOutput
    }

    func evaluate(_ sample: MNISTSample) -> [Float] {
        let input = inputLayer.forward(input: sample.image)
        let hidden = hiddenLayer.forward(input: input)
        let output = outputLayer.forward(input: hidden)
        return output
    }
}

struct OutOfBoundsError: Error {}
func trapOutOfBounds<F: BinaryFloatingPoint>(_ num: F, bounds: ClosedRange<F> = -1...1) throws {
    if !bounds.contains(num) {
//        throw OutOfBoundsError()
    }
}

func trapOutOfBounds<F: BinaryFloatingPoint>(_ array: [F], bounds: ClosedRange<F> = -1...1) throws {
    for e in array { try trapOutOfBounds(e, bounds: bounds) }
}

func trapOutOfBounds<F: BinaryFloatingPoint>(_ array: [[F]], bounds: ClosedRange<F> = -1...1) throws {
    for a in array {
        try trapOutOfBounds(a, bounds: bounds)
    }
}
