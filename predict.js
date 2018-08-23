require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs');
const fs = require('fs');
var nj = require('numjs');

//random dist; helper function
function randn_bm(min, max, skew) {
    var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    let num = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );

    num = num / 10.0 + 0.5; // Translate to 0 -> 1
    if (num > 1 || num < 0) num = randn_bm(min, max, skew); // resample between 0 and 1 if out of range
    num = Math.pow(num, skew); // Skew
    num *= max - min; // Stretch to fill range
    num += min; // offset to min
    return num;
}

async function predict() {
  const model = await tf.loadModel('file://model.json');

  batch_size = 64
  noise = nj.zeros([batch_size, 1, 1, 100])
  for (var i = 0; i < batch_size; i++) {
    for (var j = 0; j < 100; j++) {
      noise.set(i, 0, 0, j, randn_bm(-1, 1, 1))
    }
  }

  noise_tensor = tf.tensor4d(noise.tolist())
  noise_tensor.print(true)

  generated_images = model.predict(noise_tensor) //error here
}

predict()
