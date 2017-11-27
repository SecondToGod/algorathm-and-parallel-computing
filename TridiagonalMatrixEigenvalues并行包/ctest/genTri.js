var fs = require('fs');
var os = require('os');
var dim = 50;
var wopt = {
	flag: "w",
	encoding: "utf8",
	mode: 0666
};
var stream = fs.createWriteStream('./data1.txt',wopt);

for(var i=0;i<dim;i++){
	stream.write(4 +' ');	
}
stream.write(os.EOL);
for(var i=0;i<dim-1;i++){
	stream.write(1 +' ');	
}
