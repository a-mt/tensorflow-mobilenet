require("isomorphic-fetch");

// TODO deploy on heroku

//+--------------------------------------------------------
//| TENSORFLOW PREDICTION
//+--------------------------------------------------------

// Load tensorflow
// Doc: https://js.tensorflow.org/api/0.12.5/
// Example based on https://github.com/tensorflow/tfjs-examples/blob/master/mobilenet/index.js

var tf = require('@tensorflow/tfjs');
         require('@tensorflow/tfjs-node');

// Configurations
const MODEL            = [],
      IMAGE_SIZE       = 224,
      TOPK_PREDICTIONS = 5,
      offset           = tf.scalar(127.5);

/**
 * @return Promise → tf.Model - Tensorflow Model
 */
async function loadModel() {
  if(!MODEL.length) {
    var model = await tf.loadModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

    // Warmup the model (makes the first prediction faster)
    // Call `dispose` to release the WebGL memory allocated for the return value of `predict`.
    model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

    var classes = require(__dirname + '/models/mobilenet_classes.js');

    MODEL[0] = model;
    MODEL[1] = classes;
  }
  return MODEL;
}
loadModel();

/**
 * @param Uint8Array - Image
 * @return array     - Predictions
 */
async function predict(imageData) {
  var [model, classes] = await loadModel(),
      logits = tf.tidy(() => {

    // Create batch
    var batch = tf.tensor(imageData).toFloat()            // Get tensor from Uint8Array image
                .sub(offset).div(offset)                  // Normalize from [0, 255] to [-1, 1]
                .reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]); // Reshape to a single-element batch

    // Make a prediction
    return model.predict(batch);
  });

  // Retrieve the label for the top predictions
  return getTopClasses(logits, classes);
}

/**
 * @param Promise → tf.Tensor
 * @return array
 */
async function getTopClasses(logits, classes){
  var values = await logits.data();

  // Sort predictions
  var sorted = [];
  for(let i = 0; i < values.length; i++) {
    sorted.push({ i: i, value: values[i] });
  }
  sorted.sort((a, b) => {
    return b.value - a.value;
  });

  // Get top predictions
  var n   = (sorted.length > TOPK_PREDICTIONS ? TOPK_PREDICTIONS : sorted.length),
      res = [];

  for(let i = 0; i < n; i++) {
    var k = sorted[i];

    if(k.value < 0.005) {
      continue;
    }
    res.push({
      label: classes[k.i],
      value: k.value
    });
  }
  return res;
}

//+--------------------------------------------------------
//| CONFIGURE WEB SERVER
//+--------------------------------------------------------

// Configure server
var express = require('express'),
    session = require('express-session'),
    app     = express(),

    fs      = require('fs'),
    upload  = require('multer')(),
    sharp   = require('sharp');

app.use(session({
    secret: 'ce0c04361d',
    resave: true,
    saveUninitialized: true,
    cookie: {maxAge: 24 * 60 * 60 * 1000} // 1 day
}));

app.use('/public', express.static(__dirname + '/public'));
app.set('views', __dirname);
app.set('view engine', 'pug');
app.locals.pretty = true;

// Handle requests
app.post('*', upload.single('file'), function(req, res){

  var render = (e, predicted, imgPath) => {
    req.session.err = e;
    req.session.predicted = predicted;
    req.session.img = imgPath;
    res.redirect("/");
  };

  // Uploaded file
  if(req.file) {
    if(!req.file.mimetype.match("^image/")) {
      render("Unexpected file format (" + req.file.mimetype + ")");

    } else {
      doPredict(req.file.buffer, req.sessionID, render);
    }

  // Example image
  } else if(req.body.img) {
    var filepath = __dirname + '/public/' + req.body.img + '.jpg';

    if(!fs.existsSync(filepath)) {
      render("File \"" + req.body.img + "\" doesn't exist");

    } else {
      fs.readFile(filepath, function(e, buffer) {
        if(e) {
          render(e);
        } else {
          doPredict(buffer, req.sessionID, render);
        }
      });
    }

  // None ??
  } else {
    render();
  }
});

app.get('*', function(req, res){
  var err       = req.session.err || null,
      predicted = req.session.predicted || null,
      imgPath   = req.session.img || null,
      img;

  delete req.session.err;
  delete req.session.predicted;
  delete req.session.img;

  if(imgPath && fs.existsSync(imgPath)) {
    img = new Buffer(fs.readFileSync(imgPath)).toString('base64');
    fs.unlink(imgPath, () => {});
  }

  res.render('index', { err, predicted, img });
});

/**
 * @param Buffer buffer
 * @param function render - (e, predicted)
 */
function doPredict(buffer, sessionID, render) {
  var tmpSave = '/tmp/' + sessionID + '.jpg';

  // Prepare image
  // Doc: http://sharp.pixelplumbing.com/en/stable/
  var transform = sharp(buffer)
    .resize(IMAGE_SIZE, IMAGE_SIZE, { kernel: sharp.kernel.nearest }) // Resize (preserves ratio)
    .background('white').embed().flatten()                            // Pad with white
    .toFormat(sharp.format.jpeg);                                     // Convert to JPEG

  // Save tmp image
  transform.clone().toFile(tmpSave);

  // Retrieve raw RGB and predict content
  transform.raw().toBuffer(function(e, outputBuffer) {
      if(e) { return render(e.message); }

      predict(new Uint8Array(outputBuffer)).then((list) => {
        render(null, list, tmpSave);
      }).catch((e) => {
        render(e.message);
      });
    });
}


//+--------------------------------------------------------
//| START WEB SERVER
//+--------------------------------------------------------
var port = process.env.PORT || 8080;
app.listen(port, function(){
   console.log('The server is listening on port ' + port); 
});