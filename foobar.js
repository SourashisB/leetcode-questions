example1 = ["foobarhellobarfoo"]
blag = ['foo', 'bar', 'hello', 'bar', 'foo']
example2 = ["foobyehihello"]
example1list = ["foo", "bar"]
example2list = ["foo", "bye", "bye"]

function checkString(words, substring) {
    v1 = substring[0];

    const expression = new RegExp(v1, "g");
    
    const matches = words[0].match(expression)

    return matches
    
    
}

console.log(checkString(example1, example1list))