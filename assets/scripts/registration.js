/**
 * Created by pranjal on 28-01-2021.
 */
angular.module('webapp', ['ngMaterial'])
    .controller('AppCtrl', function($scope) {
        $scope.str_username = '' ;
        $scope.str_email = '' ;
        $scope.str_company = '' ;

        $scope.answer = function()
        {
            if($scope.str_username !='' ||$scope.str_username != null || $scope.str_email!= '' || $scope.str_email != null || $scope.str_company != '' || $scope.str_company != null)
            {
                if(validate_email($scope.str_email))
                {
                    window.api.send('credentials',{
                        username : $scope.str_username,
                        email : $scope.str_email,
                        company : $scope.str_company
                    });
                }
            }
        };

        function validate_email(email)
        {
            const re = /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
            return re.test(String(email).toLowerCase());
        }


    });