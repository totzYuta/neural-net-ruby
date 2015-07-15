#!/usr/bin/env ruby

require_relative '../neural_net'

# This neural network will predict the character '0' or '1'

rows = File.readlines("examples/number_image.data").map {|l| l.chomp.split(',') }

# rows.shuffle!

label_encodings = {
  0 => [1, 0],
  1 => [0, 1]
}

x_data = rows.map {|row| row[0,16].map(&:to_i) }
y_data = rows.map {|row| label_encodings[row[17].to_i] }

# Training Data
x_train = x_data.slice(0, 4)
y_train = y_data.slice(0, 4)

# Testing Data
x_test = x_data.slice(4, 1)
y_test = y_data.slice(4, 1)

# Build a 3 layer network: 16 input neurons, 8 hidden neurons, 2 output neurons
# Bias neurons are automatically added to input + hidden layers; no need to specify these
nn = NeuralNet.new [16,8,2]

prediction_success = -> (actual, ideal) {
  predicted = (0..1).max_by {|i| actual[i] }
  ideal[predicted] == 1 
}

mse = -> (actual, ideal) {
  errors = actual.zip(ideal).map {|a, i| a - i }
  (errors.inject(0) {|sum, err| sum += err**2}) / errors.length.to_f
}

error_rate = -> (errors, total) { ((errors / total.to_f) * 100).round }

run_test = -> (nn, inputs, expected_outputs) {
  success, failure, errsum = 0,0,0
  inputs.each.with_index do |input, i|
    output = nn.run input
    prediction_success.(output, expected_outputs[i]) ? success += 1 : failure += 1
    errsum += mse.(output, expected_outputs[i])
  end
  [success, failure, errsum / inputs.length.to_f]
}

# puts "Testing the untrained network..."

# success, failure, avg_mse = run_test.(nn, x_test, y_test)

# puts "Untrained classification success: #{success}, failure: #{failure} (classification error: #{error_rate.(failure, x_test.length)}%, mse: #{(avg_mse * 100).round(2)}%)"


puts "\nTraining the network...\n\n"

t1 = Time.now
result = nn.train(x_train, y_train, error_threshold: 0.01, 
                                    max_iterations: 100,
                                    log_every: 10
                                    )

# puts result
puts "\nDone training the network: #{result[:iterations]} iterations, #{(result[:error] * 100).round(2)}% mse, #{(Time.now - t1).round(1)}s"


puts "\nTesting the trained network..."

success, failure, avg_mse = run_test.(nn, x_test, y_test)
puts "Trained classification success: #{success}, failure: #{failure} (classification error: #{error_rate.(failure, x_test.length)}%, mse: #{(avg_mse * 100).round(2)}%)"
