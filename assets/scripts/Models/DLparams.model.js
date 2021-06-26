/**
 * Created by pranjal on 15-09-2020.
 */
const mongoose = require('mongoose');

var NN_Schema = new mongoose.Schema({
    model_id : {type : String},

    fliename : {
                type : String,
                required : true },
    ptype : {type : String},

    layers : { type : Number},
    layers_name : { type : Array},
    neurons : { type :Array},
    dropouts : { type : Array },

    weights : {type : Array},
    biases : { type : Array },
    hidden_weights : { type : Array },
    hidden_biases : { type : Array },

    framework : {type : String},
    test_split : { type : Number },
    validation_split : { type : Number },
    learning_rate : { type : Number },
    Batch_size : { type : Number },
    Optimizer :  { type : String  },
    date : { type : Date  },
    epochs : { type : Number },

    test_accuracy : { type : Number },
    train_accuracy : { type : Number },
    train_loss : { type : Number },
    val_loss : { type : Number }

});

var User_info = new mongoose.Schema({
   UserName : {type : String,
                required : true
   },

  /*  Id : {type : String,
        unique : true
    },

    Name : {type : String,
        required : true
    },

    image : {type : String,
        required : true
    },*/
    NeuralNet : { type : [NN_Schema]}
});


//mongoose.model('Neural_Model',NN_Schema);
mongoose.model('User',User_info);