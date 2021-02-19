// All of the Node.js APIs are available in the preload process.
// It has the same sandbox as a Chrome extension.

const { contextBridge, ipcRenderer } = require('electron') ;
const fs = require('fs');



window.addEventListener('DOMContentLoaded', () => {
  const replaceText = (selector, text) => {
    const element = document.getElementById(selector)
    if (element) element.innerText = text
  }

  for (const type of ['chrome', 'node', 'electron']) {
    replaceText(`${type}-version`, process.versions[type])
  }
})


/*
process.once('loaded',function( ){
  window.addEventListener('message',)

}) */

contextBridge.exposeInMainWorld(
    "api", {
      send : function(channel,data) {
        //const validchannels = ["toMain", "headers" , "params"] ;
        const validchannels = ["toMain", "paramstoMain", 'killscript', 'restart', 'deletedoc', 'history', 'credentials'] ;
        if(validchannels.includes(channel))
        {
          ipcRenderer.send(channel,data);
        }

      },
      receive : function(channel,func){
        const validchannels = ["fromMain", "Input_Features", "Display_Message", "Result", "Code", 'history', 'registered', 'restarted', 'deleted'] ;
        if(validchannels.includes(channel))
        {
          //console.log(data);
          ipcRenderer.on(channel, function(event,args){
          func(args);
          });
          //return data ;
        }
      }
    }
);


