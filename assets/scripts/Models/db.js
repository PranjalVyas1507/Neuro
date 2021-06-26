/**
 * Created by pranjal on 11-09-2020.
 */

const mongoose = require('mongoose');

mongoose.connect('mongodb://3.6.176.123:27017/Models',{
    auth : {
        user: 'NeuroUser',
        password: 'Neurover1',
        authdb:"admin"
    },
    authSource :"admin",
    useNewUrlParser : true,
    useUnifiedTopology : true
},function(err){
    if(!err){
        console.log('connected to the AI params database');
    }
    else
    {
        console.log('Error connecting to database');
        console.log(err);
    }
});

require('./DLparams.model');