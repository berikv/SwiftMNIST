import Foundation

@Observable
@MainActor
final class NeuralNetworkTrainer {
    let batchSize = 100

    private(set) var isTrainging = false
    private(set) var startDate: Date?
    private(set) var stopDate: Date?
    private(set) var error = [Double]()
    private(set) var task: Task<(), any Error>?

    func start() {

        stopDate = nil
        startDate = Date()
        error.removeAll(keepingCapacity: true)
        isTrainging = true

        task = Task.detached { [unowned self] in
            let engine = NeuralEngine()
            var errors = [Double]()
            var batch = [Double]()
            batch.reserveCapacity(batchSize)

            for (index, sample) in MNISTDataset.training.enumerated() {
                let prediction = engine.evalute(sample)
                let error = meanSquaredError(prediction: prediction, target: sample.target)
                
                errors.append(error)
                batch.append(errors.mean)

                if index.isMultiple(of: batchSize) {
                    Task { @MainActor [batch] in updateError(batch: batch) }
                    batch.removeAll(keepingCapacity: true)
                    try Task.checkCancellation()
                }
            }

            Task { @MainActor [batch] in updateError(batch: batch) }
            await finishTask()
        }
    }

    func stop() {
        guard isTrainging else { return }
        task?.cancel()
        finishTask()
    }

    private func updateError(batch: [Double]) {
        error.append(contentsOf: batch)
    }

    private func finishTask() {
        stopDate = Date()
        task = nil
        isTrainging = false
    }
}
