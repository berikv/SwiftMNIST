import SwiftUI

import CoreGraphics

struct ContentView: View {
    @State private var selected = 3
    @State private var training = MNISTDataset.training
    @State private var validation = MNISTDataset.validation

    var body: some View {
        NavigationView {
            List(selection: $selected) {
                NavigationLink("View training set") {
                    MNISTDatasetView(dataset: $training)
                        .padding()
                }.tag(0)

                NavigationLink("View validation set") {
                    MNISTDatasetView(dataset: $validation)
                        .padding()
                }.tag(1)

                NavigationLink("Brain engine") {
                    BrainEngineView()
                        .padding()
                }.tag(2)

                NavigationLink("NN Engine") {
                    NeuralNetworkEnigenView()
                        .padding()
                }.tag(3)
            }.padding()
        }
    }
}


#Preview {
    ContentView()
}
