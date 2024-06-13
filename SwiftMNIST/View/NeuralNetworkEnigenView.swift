import SwiftUI
import Charts

@MainActor
struct NeuralNetworkEnigenView: View {
    let trainer = NeuralNetworkTrainer()

    var body: some View {
        Chart {
            ForEach(trainer.error.indices) { index in
                // Chart doesn't like it when there suddenly aren't any indices anymore
                // Even with a range of 0..<0 it'll still try to create a LineMark with index 0
                // which causes an index out of bound error. Using a ternary here to work around that.
                LineMark(
                    x: .value("index", index),
                    y: .value("error", index < trainer.error.endIndex ? trainer.error[index] : -1))
            }
        }

        VStack {
            HStack {
                Button("Train") {
                    trainer.start()
                }.disabled(trainer.isTrainging)

                Button("Stop") {
                    trainer.stop()
                }.disabled(!trainer.isTrainging)
            }

            if let trainingStart = trainer.startDate, let trainingStop = trainer.stopDate {
                VStack {
                    let rate = Double(trainer.error.count) / trainingStop.timeIntervalSince(trainingStart)
                    Text("Training took \(relative: trainingStart...trainingStop) " + "at \(rate, specifier: "%.0f") samples per second")
                    Text("Processed \(trainer.error.count) samples in total")
                }
            } else if let trainingStart = trainer.startDate {
                let rate = Double(trainer.error.count) / Date.now.timeIntervalSince(trainingStart)
                Text("Training for \(trainingStart, style: .relative) at \(rate, specifier: "%.0f") samples per second")
            }
        }
    }
}

#Preview {
    NeuralNetworkEnigenView()
}
