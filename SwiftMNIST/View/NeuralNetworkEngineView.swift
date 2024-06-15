import SwiftUI
import Charts

@MainActor
struct NeuralNetworkEngineView: View {
    let trainer = NeuralNetworkTrainer()

    var body: some View {
        Chart {
            ForEach(trainer.trainingMSE.indices) { index in
                // Chart doesn't like it when there suddenly aren't any indices anymore
                // Even with a range of 0..<0 it'll still try to create a LineMark with index 0
                // which causes an index out of bound error. Using a ternary here to work around that.
                LineMark(
                    x: .value("index", index),
                    y: .value("error", index < trainer.trainingMSE.endIndex ? trainer.trainingMSE[index] : -1))
            }
        }

        Chart {
            ForEach(trainer.validationResults.indices) { index in
                // Chart doesn't like it when there suddenly aren't any indices anymore
                // Even with a range of 0..<0 it'll still try to create a LineMark with index 0
                // which causes an index out of bound error. Using a ternary here to work around that.
                LineMark(
                    x: .value("index", index),
                    y: .value("error", index < trainer.validationResults.endIndex ? trainer.validationResults[index].meanSquaredError : -1))
            }
        }

        VStack {
            HStack {
                Button("Train") {
                    trainer.train()
                }.disabled(trainer.isTraining)
                    .onChange(of: trainer.isTraining) { wasTraining, isTraining in
                        if wasTraining && !isTraining {
                            trainer.validate()
                        }
                    }

                Button("Validate") {
                    trainer.validate()
                }.disabled(trainer.isValidating)

                Button("Stop") {
                    trainer.stop()
                }.disabled(!trainer.isTraining && !trainer.isValidating)
            }

            VStack {
                if trainer.isTraining && trainer.isValidating {
                    Text("Training and Validating")
                } else if trainer.isTraining {
                    Text("Training")
                } else if trainer.isValidating {
                    Text("Validating")
                } else {
                    Text("Idle")
                }

                if let trainingStart = trainer.startDate, let trainingStop = trainer.stopDate {
                    let rate = Double(trainer.trainingMSE.count) / trainingStop.timeIntervalSince(trainingStart)
                    Text("Training took \(relative: trainingStart...trainingStop) " + "at \(rate, specifier: "%.0f") samples per second")
                    Text("Processed \(trainer.trainingMSE.count) samples in total")
                    Text("Error \(trainer.trainingMSE.mean)")
                } else if let trainingStart = trainer.startDate {
                    let rate = Double(trainer.trainingMSE.count) / Date.now.timeIntervalSince(trainingStart)
                    Text("Training for \(trainingStart, style: .relative) at \(rate, specifier: "%.0f") samples per second")
                    Text("Error \(trainer.trainingMSE.mean)")
                }

                if !trainer.validationResults.isEmpty {
                    let mseMean = trainer.validationResults.map { $0.meanSquaredError }.mean
                    let correctCount = trainer.validationResults.filter { $0.isCorrect }.count
                    let totalCount = trainer.validationResults.count
                    let correctPct = Double(correctCount) / Double(totalCount) * 100
                    Text("Validation mean MSE \(mseMean), correct \(correctPct, specifier: "%.0f")%, \(correctCount)/\(totalCount)")
                }
            }
        }
    }
}

#Preview {
    NeuralNetworkEngineView()
}
