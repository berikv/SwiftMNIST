import Foundation

@Observable
@MainActor
final class NeuralNetworkTrainer {
    private(set) var startDate: Date?
    private(set) var stopDate: Date?
    private(set) var trainingMSE = [Float]()
    private(set) var validationResults = [ValidationResult]()
    private var trainTask: Task<(), any Error>?
    private var validateTask: Task<(), any Error>?

    var isTraining: Bool { trainTask != nil }
    var isValidating: Bool { validateTask != nil }

    private let batchSize = 500
    private var engine = NeuralEngine()

    func train() {
        guard trainTask == nil else { return }
        
        stopDate = nil
        startDate = Date()
        trainingMSE.removeAll(keepingCapacity: true)

        trainTask = Task.detached { [unowned self] in
            var engine = await self.engine
            var errors = [Float]()
            var batch = [Float]()
            batch.reserveCapacity(batchSize)

            for (index, sample) in MNISTDataset.training.shuffled().enumerated() {
                let prediction = engine.train(sample)
                let error = meanSquaredError(prediction: prediction, target: sample.target)
                
                errors.append(error)
                batch.append(errors.mean)

                if index.isMultiple(of: batchSize) {
                    await updateTraining(engine: engine, errorBatch: batch)
                    batch.removeAll(keepingCapacity: true)
                    try Task.checkCancellation()
                }
            }

            await updateTraining(engine: engine, errorBatch: batch)
            await finalizeTraining()
        }
    }

    func stop() {
        guard let trainTask else { return }
        trainTask.cancel()
        finalizeTraining()
    }

    struct ValidationResult {
        let sample: MNISTSample
        let prediction: [Float]
        let meanSquaredError: Float
        var isCorrect: Bool {
            let mostLikelyPredicted = prediction.firstIndex(of: prediction.max()!)
            return mostLikelyPredicted == Int(sample.label)
        }
    }

    func validate() {
        validateTask = Task.detached { [unowned self] in
            let engine = await self.engine
            var results = [ValidationResult]()
            for (index, sample) in MNISTDataset.validation.enumerated() {
                let prediction = engine.evaluate(sample)
                let mse = meanSquaredError(prediction: prediction, target: sample.target)

                results.append(ValidationResult(
                    sample: sample,
                    prediction: prediction,
                    meanSquaredError: mse))
            }

            Task { @MainActor [results] in
                validationResults = results
                validateTask = nil
            }
        }
    }

    func reset() {
        trainTask?.cancel()
        validateTask?.cancel()

        engine = NeuralEngine()
    }

    private func updateTraining(engine: NeuralEngine, errorBatch: [Float]) {
        self.engine = engine
        trainingMSE.append(contentsOf: errorBatch)
    }

    private func finalizeTraining() {
        stopDate = Date()
        trainTask = nil

        validate()
    }
}
