import SwiftUI

import CoreGraphics

struct ContentView: View {
    @State var dataset = MNISTDataset.training
    
    var body: some View {
        VStack {
            MNISTDatasetView(dataset: $dataset)
        }
        .padding()
    }
}


#Preview {
    ContentView()
}
