// Modules to control application life and create native browser window
const {app, BrowserWindow, ipcMain, screen, dialog} = require('electron');
const nativeImage = require('electron').nativeImage ;
const path = require('path');
const { PythonShell } = require('python-shell');
var fs = require('fs');
const mongoose = require('mongoose');
const util = require('util') ;

require('./assets/scripts/Models/db');
const user = mongoose.model('User');
const readdir = util.promisify(fs.readdir) ;

var options = {
  //args : ['args'],
  //pythonOptions : ['-u '],
  scriptPath : path.join(__dirname,'/../Neuro/DLengine')
};



var pyscript = new PythonShell('main.py',options) ;
//var DScheck = new PythonShell('inspectDS.py',DSoptions);
var mainWindow, regWindow, NNWindow ;

//var returnobj = null ;    // object returned from python
var params = {}, credentials = {}, weights = {}, result = {}, NeuralNet_history = {} ;
var output, gen_code, code ;
var datasetcase = 0 ;
var train_contents, test_contents, val_contents ;

//ipc and python flags
var pyreturnflag = false ;
var apprequestflag = false, code_generated = false ;

function createWindow () {
  // Create the browser window.
   const { width, height } = screen.getPrimaryDisplay().workAreaSize ;
   //console.log(width,height);
   mainWindow = new BrowserWindow({
    width: width,
    height: height,
    webPreferences: {
      contextIsolation : true,
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration : false,
      enableRemoteModule : false
    },
    autoHideMenuBar : true,
    icon : nativeImage.createFromPath(__dirname + "/assets/images/NeuroIcon.png")

  });

  // and load the index.html of the app.
  mainWindow.loadFile('index.html');

  // Open the DevTools.
  // mainWindow.webContents.openDevTools()
}

function createRegistrationWindow() {
    // Create Registration Window
    regWindow =  new BrowserWindow({
        width : 400,
        height : 500,
        webPreferences : {
            contextIsolation : true,
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration : false,
            enableRemoteModule : false
        },
        movable : false,
        minimizable : false,
        maximizable : false,
        autoHideMenuBar : true,
        icon : nativeImage.createFromPath(__dirname + "/assets/images/NeuroIcon.svg")

    }) ;

    regWindow.loadFile('assets/html/Registration.html')
}


function createPlaygroundWindow()
{
    const { width, height } = screen.getPrimaryDisplay().workAreaSize ;
    NNWindow =  new BrowserWindow({
        width: width,
        height: height,
        webPreferences: {
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: false,
            enableRemoteModule: false
        },
        movable: false,
        minimizable: false,
        maximizable: false,
        autoHideMenuBar: true,
        icon: nativeImage.createFromPath(__dirname + "/assets/images/NeuroIcon.svg")
    }) ;

    NNWindow.loadFile('assets/html/NNPark.html') ;
    };



function startNeuro()
{
    if(fs.existsSync('user.json'))
    {
        createWindow() ;
    }
    else
    {
        createRegistrationWindow() ;
    }
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  /*createWindow();
  createRegistrationWindow();*/
  startNeuro() ;

  app.on('activate', function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) /*createWindow()*/ startNeuro() ;
  })
});

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', function () {
  if (process.platform !== 'darwin')
  {
      delete_temp_files() ;
      app.quit()
  }
}) ;

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.

ipcMain.on("toMain",function(event,args){
    //console.log(args);
  /*var options = {
    args : [args],
    pythonOptions : [ '-u ' ]
  };

  //pyscript = new PythonShell('main.py',options); */
  //appreturnflag = false ;
  console.log("to-main called upload button clicked");
  apprequestflag = true ;
    fs.writeFile(path.resolve(__dirname,'DLengine/path.txt'),JSON.stringify(args),{flag : 'w' },function(err,results)
    {
        if(err)
        {
            console.log('error',err);
        }
        else console.log('result',results) ;
    });
  /*pyscript.send(args).end(function(err){
      if (err)
      {
          console.log(err);
      }
  });*/
});

ipcMain.on("paramstoMain",function(event,args){
  console.log("hyperparameters received");
  console.log(args);
  params = args ;
  fs.writeFile(path.resolve(__dirname,'DLengine/params.json'),JSON.stringify(args),{flag : 'w' },function(err,results)
    {
        if(err)
        {
            console.log('error',err);
        }
    });

    /*fs.readFile(path.resolve(__dirname,'DLengine/params.json'),"utf8",function(error,result)
    {
        console.log(result);
    })*/

});

/*ipcMain.on("params",function(event,args){
  pyscript.send(args);
}); */

ipcMain.on("killscript",function(event,args)
{
    //pyscript.kill('SIGTERM') ;
    console.log("restart script");
    delete_temp_files();
    if(args == 'restart')
    {
        fs.writeFile(path.resolve(__dirname,'DLengine/reset.json'),JSON.stringify(args),{flag : 'w' },function(err,results)
        {
            if(err)
            {
                console.log('error',err);
            }
        });

    }

    if(args === 'reupload')
    {

    }
});

ipcMain.on('deletedoc',function(event,args){
    var file_name = args.file ;
    var date = args.date;
    console.log("In delete doc");
    user.updateOne(
        { UserName : getNeuroUser() },
        { $pull : { NeuralNet : { fliename : file_name }  } },function(error , success){
            if(error)
            {
                console.log('Error:\t' + error);
            }
            else
            {
                console.log(success);
            }
        }
    );

    var filename = getNeuroUser() + '.json';
    //console.log(filename);
    var result = JSON.parse(fs.readFileSync(filename, 'utf8'));
    if('NeuralNet' in result)
    {
        //var i;
        var model_date, deleted = false ;
        //console.log(Object.keys(result['NeuralNet']).length);
        for(i=0;i<Object.keys(result['NeuralNet']).length;++i)
        {
            model_date = new Date(result['NeuralNet'][i].date);
            if((result['NeuralNet'][i].fliename === file_name)&&(model_date.toString() === date.toString()))
            {
                console.log("Matched condition");
                result['NeuralNet'].splice(i,1);
                fs.writeFile(path.resolve(__dirname,filename),JSON.stringify(result),{flag : 'w' },function(err,results)
                {
                    if(err)
                    {
                        console.log('error',err);
                    }
                    else
                    {
                        console.log(results);
                    }
                });
                deleted = true
            }
        }

        if(deleted === true)
        {
            mainWindow.webContents.send('deleted',true);
            deleted = false;
        }
        else
        {
            mainWindow.webContents.send('deleted',false);
        }

    }

});


ipcMain.on('credentials',function(event,args)
{
    console.log(args) ;
    credentials = args ;
    newuser = {
        UserName : credentials.username,
        Email : credentials.email,
        Company : credentials.company
    }
    fs.writeFile(path.resolve(__dirname,'user.json'),JSON.stringify(args),{flag : 'w' },function(err,results)
    {
        if(err)
        {
            console.log('error',err);
        }
        else
        {
            console.log(results);
            User = user.create(newuser) ;
        }
    });
    var filename = args.username + '.json' ;
    if(fs.existsSync("filename"))
    {
        console.log("User already exists");
    }
    else
    {
        fs.writeFile(path.resolve(__dirname,filename),JSON.stringify(args),{flag : 'w' },function(err,results)
        {
            if(err)
            {
                console.log('error',err);
            }
            else
            {
                console.log(results);
                //User = user.create(newuser) ;
            }
        });
        createWindow() ;
        regWindow.close() ;
    }

});

ipcMain.on('online-status-changed', function(event, status) {
    console.log(status)
});

ipcMain.on('history',function(event, args) {
    //console.log(args);
    //console.log(typeof getNeuroUser())
    if(args === true)
    {
       // async_retrieve();
        /*user.findOne({UserName: getNeuroUser()}, function (err, docs) {
            //NeuralNet_history = docs.NeuralNet;
            //console.log(docs);
            for (i = 0; i < docs.NeuralNet.length; ++i) {
                NeuralNet_history['file' + (i + 1)] = docs.NeuralNet[i];
            }
        });*/
        var filename = getNeuroUser() + '.json';
        //console.log(filename);
        var result = JSON.parse(fs.readFileSync(filename, 'utf8'));
        if('NeuralNet' in result)
        {
            NeuralNet_history = result['NeuralNet'];
            //console.log(NeuralNet_history);
            mainWindow.webContents.send('history',JSON.stringify(NeuralNet_history));
        }
            }
});


ipcMain.on('AddDatabase',function(event, args){
    (async() =>{
       //console.log(await opendialog((args))) ;
        var results = await opendialog((args));
        //console.log(results) ;
        if(results.canceled == false)
        {
            get_databasecase(results.filePaths[0]);
            console.log("datacase:\t"+dataset_case);
        }
    })() ;
});

ipcMain.on("OpenNNPark", function(event,args){
    createPlaygroundWindow();
});



pyscript.on('message',function(message){
     //console.log(message);
    pyreturnflag = true;
    var i ;
    //console.log("Message data type: ",typeof message)
    message = message.split("\n");
    console.log(message);
    //console.log(message.length);
    if(message.length>1)
    {
        for(i=0;i<message.length;++i)
        {
            if(message[i].includes("Notification"))
            {
                console.log("Contains Notification");
                if(message[i].includes("Input_Features"))
                {
                    console.log("Includes input features");
                    ip_features = message[i].split("__")[1];
                    console.log(ip_features);
                    mainWindow.webContents.send("Input_Features",ip_features);
                }
            }
        }
    }

    else
    {
        if((message[0].includes("Display_Message")) || (message[0].includes("Check Result")))
        {
           if(message[0].includes("Check Result"))
            {
                    if(fs.statSync("result.json").isFile())
                    {
                        output = fs.readFileSync('result.json', 'utf8');
                        result = JSON.parse(output);
                        mainWindow.webContents.send("Result",result);
                    }
                    if(fs.statSync("weights.json").isFile())

                    {
                        output = fs.readFileSync('weights.json', 'utf8');
                        weights = JSON.parse(output);
                        console.log("Weights:\n",weights);
                    }
                    storeparams();

            }
            message = message[0].split(":",2);
             mainWindow.webContents.send("fromMain",message[1]);
        }

        else if((message[0].includes("Epoch")) || (message[0].includes("loss")))
        {
            console.log(message[0]);
            mainWindow.webContents.send("fromMain",message[0]);
        }
        else if(message[0].includes("Final_Message : Python-Code generated"))
        {
            code_generated = false ;
            while(code_generated === false)
            {
                    if(fs.existsSync('DL_code.py'))
                    {
                        gen_code = fs.readFileSync('DL_code.py', 'utf8');
                        code = JSON.parse(JSON.stringify(gen_code));
                        mainWindow.webContents.send("Code",code);
                        code_generated = true ;
                    }
            }

        }
        else if(message[0].includes("Notification"))
        {
            console.log("Contains Notification");
            if(message[0].includes("Input_Features"))
            {
                console.log("Includes input features");
                ip_features = message[0].split("__")[1];
                console.log(ip_features);
                mainWindow.webContents.send("Input_Features",ip_features);
            }

            if(message[0].includes("Restarted"))
            {
                fs.unlinkSync('DLengine/reset.json');
                mainWindow.webContents.send("restarted",'restarted')
            }
        }
        else if(message[0].includes("Error"))
        {
            mainWindow.webContents.send("fromMain",message[0]);
            delete_temp_files();
        }
        else if(message[0].includes("Categories"))
        {
            if(message[0].includes("name"))
            {
                message = message[0].split(":",2);
                //console.log(typeof message[1]);
                message = (message[1]);
                //console.log(typeof message);
                mainWindow.webContents.send("Category_array",message);
            }
            else
            {
                message = message[0].split(":",2);
                //console.log(typeof message[1]);
                message = Number(message[1]);
                //console.log(typeof message);
                mainWindow.webContents.send("Categories",message);
            }
        }

    }
});

pyscript.on('close',function(message)
{
    console.log(message);
});


function delete_temp_files()
{
        if(fs.existsSync("DL_code.py")) fs.unlinkSync('DL_code.py');
        if(fs.existsSync("DLengine/path.txt")) fs.unlinkSync('DLengine/path.txt');
        if(fs.existsSync("DLengine/params.json")) fs.unlinkSync('DLengine/params.json');
        if(fs.existsSync("result.json")) fs.unlinkSync('result.json');
        if(fs.existsSync("weights.json")) fs.unlinkSync('weights.json');
}

function storeparams()
{
    username = getNeuroUser();
    console.log("Storing params for",username);
    var NeuralNetworkmodel ;
    if((params.type ==='Classification')||(params.type ==='MultiClass'))
    {
        console.log(params.type);
        NeuralNetworkmodel = {
            fliename : params.filename,
            layers : params.layers ,
            neurons : params.neurons,
            ptype : params.type,
            test_split : params.testsplit,
            validation_split : params.validsplit,
            learning_rate : params.learning_rate,
            Batch_size : params.batch_size,
            Optimizer : params.optimization,
            dropouts : params.dropouts,
            framework : params.framework,
            date : Date.now(),
            layers_name : weights.layers,
            weights : weights.weights,
            biases : weights.biases,
           // hidden_weights : weights.U,
           // hidden_biases : weights.biases_hh,
            train_accuracy : result.accuracy[result.accuracy.length - 1],
            test_accuracy : result.val_accuracy[result.val_accuracy.length - 1],
            train_loss : result.loss[result.loss.length - 1],
            val_loss : result.val_loss[result.val_loss.length - 1],
            epochs : params.epochs
        };
    }
    else
    {
        //console.log("TimeSeries");
        NeuralNetworkmodel = {
            fliename : params.filename,
            layers : params.layers ,
            neurons : params.neurons,
            ptype : params.type,
            test_split : params.testsplit,
            validation_split : params.validsplit,
            learning_rate : params.learning_rate,
            Batch_size : params.batch_size,
            Optimizer : params.optimization,
            dropouts : params.dropouts,
            framework : params.framework,
            date : Date.now(),
            layers_name : weights.layers,
            weights : weights.weights,
            biases : weights.biases,
            hidden_weights : weights.U,
            hidden_biases : weights.biases_hh,
            train_loss : result.loss[result.loss.length - 1],
            val_loss : result.val_loss[result.val_loss.length - 1],
            epochs : params.epochs
        };
    }
    storelocal(NeuralNetworkmodel);
    user.updateOne(
        { UserName : username },        //UserName : credentials.username
        {$push : { NeuralNet : NeuralNetworkmodel }},
        function(error , success){
            console.log("Inside updateone func");
            if(error)
            {
                //console.log("TimeSeries");
                console.log("Error:\t",error) ;
            }
            else
            {
                console.log("Success:\t",success);
            }
        }
    )
}

function getNeuroUser()
{
    info = JSON.parse(fs.readFileSync('user.json','utf8'));
    return info.username ;
}

async function retrieve_models()
{
    docs = await user.findOne({UserName: getNeuroUser()});
        //NeuralNet_history = docs.NeuralNet;
    for (i = 0; i < docs.NeuralNet.length; ++i) {
        NeuralNet_history['file' + (i + 1)] = docs.NeuralNet[i];
    }

}

const async_retrieve = async () =>{
    await retrieve_models().then(()=>{
        //console.log("NeuralNet_history",NeuralNet_history);
        mainWindow.webContents.send('history',JSON.stringify(NeuralNet_history));
    })
};



async function opendialog(args)
{
    if(args == 'Upload_folder')
    {
        var dialog_options = {
            title : "Upload Folder",
            //defaultPath : app.getPath('Desktop'),
            properties : [ 'openDirectory' ]
        };
        const dialog_resutls = await dialog.showOpenDialog(mainWindow, dialog_options) ;
        return dialog_resutls ;
    }

    if(args == 'Upload_File')
    {

    }
}



function storelocal(NNmodel)
{
    var filename = getNeuroUser() + '.json';
    //console.log(filename);
    var result = JSON.parse(fs.readFileSync(filename, 'utf8'));
    if(!('NeuralNet' in result))
    {
        result['NeuralNet'] = []
    }
    result['NeuralNet'].push(NNmodel);

    fs.writeFile(path.resolve(__dirname,filename),JSON.stringify(result),{flag : 'w' },function(err,results)
    {
        if(err)
        {
            console.log('error',err);
        }
        else
        {
            console.log(results);
        }
    });
}


function get_databasecase(path)
{
    var directory_count = 0, image_count = 0;
    //console.log(path) ;
    fs.readdir(path,async function(err, files)
    {
        var i ;
        if(!err)
        {
            //console.log(files) ;
            for(i=0;i<files.length;++i)
            {
               if(Isfolder(path+'\\'+files[i]) === true)
               {
                   directory_count = directory_count + 1 ;
               }

               else if(Isimage(path+'\\'+files[i]) === true)
                {
                    image_count = image_count + 1 ;
                }
            }
            //console.log("directory:"+directory_count) ;
            //console.log("Images" + image_count) ;

            await Promise.all([find_dataset_case(path,files,directory_count,image_count)]).then(()=>
            {
                try{
                    mainWindow.webContents.send('dblocation',{folder : path ,
                        dbcase : datasetcase});
                }catch (err){
                    console.log(err) ;
                }
            });
            //console.log("get_databasecase:\t"+ dataset_case );

        }


    });
    //console.log("Case outside callback:\t"+ dataset_case);
}

function Isimage(file)
{
    if(file.includes('.') === true)
    {
        //console.log(1);
        if(fs.existsSync(file) && fs.lstatSync(file).isFile())
        {
            var tmp = file.split('.');
            //console.log(tmp);
            if((tmp[1]==='jpeg')||(tmp[1]==='png')||(tmp[1]==='jpg'))
            {
               //console.log(tmp[1]);
                return true ;
            }
        }

        return false ;
    }

}

function PDP()
{

}

function FeatureDistribution()
{

}


function Isfolder(path)
{
    //console.log(path) ;
    if(fs.existsSync(path) && fs.lstatSync(path).isDirectory())
    {
        return true ;
    }

    return false ;
}


function substringsearch(substr ,filelist,path)
{
    var index ;
    //var filelist = await contains_folders(folder) ;
    for(index=0;index<filelist.length;++index)
    {
        if(filelist[index].includes(substr))
        {
            return {search : true,
              filename : filelist[index]
            };
        }
    }
    return false ;
}

function checkcontents(path,contents)
{
    var i/*, num_images, num_dir*/ ;
    var images = [], dir = [] ;
    //console.log(contents);
    for(i=0;i<contents.length;++i)
        {
            //console.log(contents[i]);
        if(Isfolder(path+"\\"+contents[i]) === true)
        {
            //num_dir = num_dir + 1;
            dir[i] = contents[i] ;
        }
        else if(Isimage(path+"\\"+contents[i]) === true)
        {
            //num_images = num_images + 1 ;
            images[i] = contents[i] ;
            //console.log(images[i])
        }
    }
    return { images : images,
            directories : dir
            }

}


async function find_dataset_case(path,files, num_directory, num_images)
{
    //console.log("Path fdc:\t"+path) ;
    //var datasetcase = 0 ;
    var traindir, testdir, valdir, devdir ;


    traindir = substringsearch('train',files);
    testdir = substringsearch('test',files);
    valdir = substringsearch('val',files);
    devdir = substringsearch('dev',files) ;

    if(num_directory==0)
    {
        console.log("No sub-directories\n");

        if(num_images<50)
        {
            datasetcase = 0 ;
        }
        else
        {
            datasetcase = 1 ;
        }
    }

    else if((traindir.search === true)&&(Isfolder(path+'\\'+traindir.filename) === true))
    {
        var valc1 ;
        datasetcase = 1 ;
        //contains_folders(path,files,[])

        if((testdir.search===true)&&(Isfolder(path+'\\'+testdir.filename) === true))
        {

            // --->> substringsearch(string,files,path)
            // --->> getfilepathcontaining substring
            // --->> readcontents ->> IsImage ->> IsFolder
            // --->> compare with val folder and test folder if present
            // --->> Provide dataset case (5 if subclasses-directories are present) or (6 if not present)

            await Promise.all([async_walkdir(path+"\\"+traindir.filename,'train'),async_walkdir(path+"\\"+traindir.filename,'test')]);
            datasetcase = datasetcase + 1 ;


            //console.log(train_contents);

            var trc1, ttc1 ;

            if((train_contents != null)&&(test_contents != null))
            {
                trc1 = checkcontents(path+'\\'+traindir.filename,train_contents) ;
                ttc1 = checkcontents(path+'\\'+testdir.filename,test_contents) ;
            }
            /*if(test_contents != null)
            {

            }*/

            console.log(ttc1);
            console.log(trc1);
            if((valdir.search===true)&&(Isfolder(path+'\\'+valdir.filename) === true))
            {
                datasetcase = datasetcase + 1 ;
                await Promise.all([async_walkdir(path+"\\"+valdir.filename,'val')]);

                if(val_contents != null)
                {
                    valc1 = checkcontents(path+'\\'+devdir.filename,val_contents) ;
                }
            }
            else if((devdir.search ===true)&&(Isfolder(path+'\\'+devdir.filename) === true))
            {
                datasetcase = datasetcase + 1 ;
                //val_contents = async_walkdir(path+"\\"+devdir.filename);
                await Promise.all([async_walkdir(path+"\\"+devdir.filename,'val')]);

                if(val_contents != null)
                {
                    valc1 = checkcontents(path+'\\'+devdir.filename,val_contents);
                }
            }

            if(arrequal(trc1.directories,ttc1.directories)&&(trc1.directories!=0))
            {
                console.log(trc1.directories);
                console.log(ttc1.directories);
                datasetcase = 2;
                if(arrequal(trc1.directories,valc1.directories))
                {
                    datasetcase = datasetcase + 1 ;
                }
            }
            else if(trc1.images.length>100 && ttc1.images.length>10)
            {
                datasetcase = 5;
                if(val_contents != null)
                {
                    if(valc1.images.length>10)
                    {
                        datasetcase = 6 ;
                    }
                }
            }


        }
        /*
        else if(has_valfolder(files)== true )
        {
            datasetcase = datasetcase + 1 ;
        }*/


    }

    else if(num_directory>1)
    {
            datasetcase = 4 ;
    }

    else datasetcase = 0 ;

    //console.log("find_dataset_case:\t"+ datasetcase) ;
    //return datasetcase ;
}


function arrequal(arr1, arr2)
{
    var arr_cmp = arr1.every(function(element,index)
    {
        return element === arr2[index] ;
    });

    if((arr1.length === arr2.length)&& arr_cmp)
    {
        return true ;
    }

    return false ;
}

function has_valfolder(files)
{
    /*
    if((files.includes('val'))||(files.includes('valset'))||(files.includes('val set'))||substringsearch('val',files).search==true)
    {
        return true ;
    }
    else if((files.includes('validation'))||(files.includes('validationset'))||(files.includes('validation set')))
    {
        return true ;
    }
    else if((files.includes('dev'))||(files.includes('devset'))||(files.includes('dev set'))||substringsearch('dev',files).search==true)
    {
        return true ;
    }*/

    var valdir = substringsearch('val',files);
    var devdir = substringsearch('dev',files);

    if(valdir.search===true)
    {
        return valdir.filename ;
    }
    else if(devdir.search===false)
    {
        return devdir.filename ;
    }

    return false ;
}

const async_walkdir = async(path,settype) =>{
    //console.log(path);
   var dir_contents = await walkdirectory(path) ;
        //console.log(settype);
        if(settype ==='train')
        {
            train_contents = dir_contents;
        }
        else if(settype ==='test')
        {
            test_contents = dir_contents;
        }
        else if(settype ==='val')
        {
            val_contents = dir_contents;
        }
        else if(settype ==='dev')
        {
            dev_contents = dir_contents;
        }
        //console.log("Dir Contents");
        //console.log(dir_contents);
}


async function walkdirectory(path)
{
    var contents, contents1 = [] ;
    try
    {
        //console.log(path);
        if(Isfolder(path) === true)
        {
           // console.log('isfolder');
            contents = await readdir(path);
        }
        else
        {
            return null ;
        }
    }catch(err)
    {
        console.log(err) ;
    }
    if(contents == undefined)
    {
        console.log('undefined')
    }
    else
    {
        //console.log(typeof contents1);
        contents1 = Array.from(Object.values(contents)) ;
        return contents1 ;
    }

}

process.on("uncaughtException",function(err){
    console.log(err);
});