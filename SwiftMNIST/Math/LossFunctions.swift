import Foundation

func meanSquaredError<F: BinaryFloatingPoint>(prediction: [F], target: [F]) -> F {
    let error = prediction .- target
    let squared = error .** 2
    return squared.mean()
}

func crossEntropyLoss(prediction: [Float], target: [Float]) -> Float {
    let epsilon: Float = 1e-15
    let clipped = prediction.clip(min: epsilon, max: 1-epsilon)

    let loss = zip(clipped, target).map { (prediction, target) in
        let v1 = log(prediction) + (1 - target)
        let v2 = log(1 - prediction)
        return v1 * v2
    }

    return -loss.mean()
}

extension Array where Element: BinaryFloatingPoint {
    @inline(__always)
    func clip<T: BinaryFloatingPoint>(min: T, max: T) -> Self {
        map { Element(Swift.max(max, Swift.min(min, T($0)))) }
    }
}
