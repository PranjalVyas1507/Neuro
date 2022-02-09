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



var Model_SubDoc = new mongoose.Schema({
    model_id : {type : String},
    model_name : {type : String},
    date_created : {type : Date},
    date_updated : {type : Date},
    framework_used : {type : String},
    Isdeployed : {type : Boolean},
    Isdeployed_date : {type : Date},
    metric_used : {type : [String]},
    train_metric : {type : Array},
    val_metric : {type : Array},
    train_loss : {type : Array},
    val_loss : {type : Array},
    hyperparamters : {type : [hyperparameters_Sub_SubDoc]},
    Neural_Net : {type : [NeuralArchitecture_Sub_SubDoc]},
    Learnings : {type : [Learnings_Sub_SubDoc]},
});

var Database_SubDoc = new mongoose.Schema({
    db_id : {type : String},
    db_name : {type : String},
    db_location : {type : String},
    Input_type : {type : String},
    date_updated : {type : Date},
    author : {type : String},
});

// var transform_SubDoc = new mongoose.Schema({});

var hyperparameters_Sub_SubDoc = new mongoose.Schema({
    epochs : {type : Number},
    batch_size : {type : Number},
    learning_rate : {type : Number},
    optimizer : {type : String},
    test_split : {type : Number},
    validation_split : {type : Number},
    LR_Warmup : {type : String},
    LR_Scheduler : {type : String},
    momentum : {type : Number},
});


var NeuralArchitecture_Sub_SubDoc = new mongoose.Schema({
    layers_name : {type : [String]},
    layer_type : {type : [String]},
    neurons : {type : [Number]},
    dropouts : {type : [Number]},
    activation_function : {type : [String]},
    weight_initializer : {type : [String]},
    kernel_initializer : {type : [String]},
});

var Learnings_Sub_SubDoc = new mongoose.Schema({
    weights : {type : [Number]},
    biases : {type : [Number]},
    hidden_weights : {type : [Number]},
    hidden_biases : {type : [Number]},
});



var Project_doc = new mongoose.Schema({
    project_name : {type : String, required : true},
    project_id : {type : String, required : true},
    date_created : {type : Date},
    date_modified : {type : Date},
    contributors : {type : Array},
    lead_author : {type : String},
    models : {type : [Model_SubDoc]},
    databases : {type : [Database_SubDoc]},
});




//mongoose.model('Neural_Model',NN_Schema);
mongoose.model('User',User_info);