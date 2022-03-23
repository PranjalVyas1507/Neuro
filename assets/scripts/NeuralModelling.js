/**
 * Created by pranjal on 27-07-2021.
 */
angular.module('NNapp', ['ngMaterial'])
    .controller('NMDCtrl', function($scope, $mdDialog, $mdToast, $mdSidenav) {
        var ModelSpace = new Two({
            type : Two.Types.svg,
            width:1300,
            height: 900,
            domElement :document.getElementById('NMB')
        });

        var BasicLayer = new Two({
            type : Two.Types.svg,
            width:1300,
            height: 900,
            domElement :document.getElementById('BasicLayers')
        });

        var NNLayers = new Two({
            type : Two.Types.svg,
            width:1300,
            height: 900,
            domElement :document.getElementById('NNLayers')
        });

        var Scaler = new Two({
            type : Two.Types.svg,
            width:1300,
            height: 900,
            domElement :document.getElementById('ScalingLayers')
        });

        var Pretrained = new Two({
            type : Two.Types.svg,
            width:1300,
            height: 900,
            domElement :document.getElementById('PretrainedModels')
        });

        var radialGradient = BasicLayer.makeRadialGradient(
            0, 0,
            10,
            new Two.Stop(0, 'rgba(255, 100  , 250, 1)', 1),
            new Two.Stop(1, 'rgba(0, 0, 128, 1)', 0)
        );



        /*
        var rectangle = two.makeRectangle(two.width / 8, two.height / 2, 100, 50);
        var center = two.makeCircle(two.width / 2, two.height / 2, 100);
        center.noStroke().fill = 'rgb(100, 100, 255)';
        rectangle.noStroke().fill = 'rgb(128, 128, 100)';

        var circles = [];
        for (var i = 0; i < 10; i++) {
            var circle = two.makeCircle(0, 0, 15);
            circle.noStroke().fill = 'rgb(100, 100, 255)';
            circle.offset = Math.random() * 1000;
            circle.radius = Math.random() * two.height / 4 + two.height / 4;
            circle.theta = Math.random() * Math.PI * 2;
            circles.push(circle);
        }

        two.update();

        center._renderer.elem.addEventListener('mouseover', function() {
            changeColor('rgb(255, 100, 100)');
        }, false);
        center._renderer.elem.addEventListener('mouseout', function() {
            changeColor('rgb(100, 100, 255)');
        }, false);

        two.bind('update', function(frameCount) {
            center.translation.set(two.width / 2, two.height / 2);
            center.scale = 0.5 * (Math.sin(frameCount / 30) + 1) / 2 + 0.5;
            for (var i = 0; i < circles.length; i++) {
                var circle = circles[i];
                var mag = circle.radius * Math.sin(circle.offset + frameCount / 15);
                var x = mag * Math.cos(circle.theta) + center.translation.x;
                var y = mag * Math.sin(circle.theta) + center.translation.y;
                circle.translation.set(x, y);
            }
        }).play();

        */
        populateLayers();
        updategraphics();
        updateModelSpace();

        function changeColor(color) {
            center.fill = color;
            for (var i = 0; i < circles.length; i++) {
                circles[i].fill = color;
            }
        }

        function updategraphics()
        {
            BasicLayer.update();
            NNLayers.update();
            Pretrained.update();
            Scaler.update();
        }

        function updateModelSpace()
        {
            ModelSpace.update();
        }
        function populateLayers()
        {
            populateBasicLayers();
            populateNN();
            populateScalers();
            populatePretrained();
        }

        function populateBasicLayers()
        {
            InputLayer();
            ExitLayer();
            AddLayer();
            SubtractLayer();
            MultiplyLayer();
            DivLayer();
            ExpLayer();
            ConcatLayer();

        }

        function populateNN()
        {

        }

        function populateScalers()
        {

        }

        function populatePretrained()
        {

        }

        function InputLayer()
        {
            DrawCircle(20, 35, 25, 'rgb(203, 195, 227, 0.75)');
            DrawTriangle(BasicLayer, 10,22.5,10,37.5,30,25);
            MakeText(BasicLayer, 20, 75, 'Input')
        }

        function ExitLayer()
        {

            DrawCircle(85,35,25,'gray');
        }

        function AddLayer()
        {
            DrawCircle(145,35,25,'white');
        }


        function SubtractLayer()
        {
           // DrawCircle()
        }

        function MultiplyLayer()
        {
            DrawCircle(30,10,10,'gray');
        }


        function DivLayer()
        {

        }

        function ExpLayer()
        {
            var exit = BasicLayer.makeCircle(30, 10, 10);
            exit.fill = radialGradient ;//getRandomColor();
            exit.stroke = 'black';
            exit.linewidth = 0.5;

        }

        function ConcatLayer()
        {
            var exit = BasicLayer.makeCircle(30, 10, 10);
            exit.fill = radialGradient ;//getRandomColor();
            exit.stroke = 'black';
            exit.linewidth = 0.5;
        }

        function LayerShelfEvents()
        {

        }

        function DrawCircle(x,y,radius, color)

        {
            var mul = BasicLayer.makeCircle(x, y, radius);
            mul.fill = color ;//getRandomColor();
            mul.stroke = 'black';
            mul.linewidth = 1.0;
        }
        function DrawRect(Layer ,x_center,y_center, width, height, text)
        {
            var block = Layer.makeRoundedRectangle()

        }

        function MakeText(Layer,x,y,text)
        {
            Layer.makeText(text, x, y, {size : 12 , family : "Arial"})

        }

        function DrawTriangle(Layer,x1,y1,x2,y2,x3,y3)
        {
            var line = Layer.makePath(x1,y1,x2,y2,x3,y3,true);
            line.stroke = 'rgb(255,180,120)';
        }



    });