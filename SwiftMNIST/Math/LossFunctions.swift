import Foundation

func meanSquaredError<F: BinaryFloatingPoint>(prediction: [F], target: [F]) -> F {
    let error = prediction .- target
    let squared = error .** 2
    return squared.mean
}

func meanSquaredErrorWithPenalty<F: BinaryFloatingPoint>(prediction: [F], target: [F]) -> F {
    let error = prediction .- target
    let squared = error.map { $0 * $0 }
    let mse = squared.mean
    let variancePenalty = prediction.variance
    let penaltyWeight: F = 0.2
    return mse + penaltyWeight * (1 - variancePenalty)
}
