angular.module('app', [])
  .controller('controller', function ($scope, $http) {

    $scope.init = function () {
      $scope.clusters =  [];
      $scope.numberOfEntity = 10;
      $scope.elaboration = false;
    };

    $scope.startElaboration = function () {
      $scope.elaboration = true;
      var data = {
        model: $scope.inputType,
        website: $scope.website
      };
      
      data['website'] = $scope.website || 'www.spaziodati.eu';

      $http({
        method: 'GET',
        url: '/suggest?website='+$scope.website+'&'+'model='+$scope.inputType+'&only_website=true&num_max=50',
        data: $.param(data),
        headers: {'Content-Type': 'application/x-www-form-urlencoded'}
      }).success(function (data) {
        $scope.clusters = data.output;
        $scope.elaboration = false;
      });
    };

    $scope.cleanUrl = function (url) {
      return decodeURIComponent(url).split('/').slice(-1)[0].replace(/_/g, ' ');
    };
  });
