var fs = require('fs');
var os = require('os');
var dim = 5000;
var wopt = {
	flag: "w",
	encoding: "utf8",
	mode: 0666
};
var stream = fs.createWriteStream('./data5000.txt',wopt);

for(var i=0;i<dim;i++){
	for(var j=0;j<dim;j++){
		if(j == i){
			stream.write( 4 + ' ');
		}
		else if(j == i-1 || j == i+1){
			stream.write(1 + ' ');
		}
		else{
			stream.write(0 + ' ');
		}
	}
	stream.write(os.EOL);
}
