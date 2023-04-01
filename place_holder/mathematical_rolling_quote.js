var quotes=[
    ["Pure mathematics is, in its way, the poetry of logical ideas.", "- Albert Einstein"],
    ["Mathematics is the music of reason.", "- James Joseph Sylvester"],
    ["Mathematics is the art of giving the same name to different things.", "- Henri Poincare"],
    ["Mathematics is the queen of sciences and number theory is the queen of mathematics.","- Carl Friedrich Gauss"],
    ["Mathematics is the key and door to the sciences.", "- Galileo Galilei"],
    ["Pure mathematics is, in its way, the poetry of logical ideas.", "-Albert Einstein"],
    ["Mathematics is the language in which God has written the universe.", "-Galileo Galilei"],
    ["The book of nature is written in the language of mathematics.", "-Galileo Galilei"],
    ["Mathematics is the queen of the sciences.", "-Carl Friedrich Gauss"],
    ["God made the integers, all else is the work of man.", "-Leopold Kronecker"],
    ["In mathematics, the art of proposing a question must be held of higher value than solving it.", "-Georg Cantor"],
    ["The essence of mathematics lies in its freedom.", "-Georg Cantor"],
    ["Mathematics is not about numbers, equations, computations, or algorithms: it is about understanding.", "-William Paul Thurston"],
    ["Mathematics is the most beautiful and most powerful creation of the human spirit.", "-Stefan Banach"],
    ["There is no branch of mathematics, however abstract, which may not someday be applied to phenomena of the real world.", "-Nikolai Ivanovich Lobachevsky"],
    ["The mathematician does not study pure mathematics because it is useful; he studies it because he delights in it and he delights in it because it is beautiful.", "-Henri Poincaré"],
    ["The only way to learn mathematics is to do mathematics.", "-Paul Halmos"],
    ["Mathematics knows no races or geographic boundaries; for mathematics, the cultural world is one country.", "-David Hilbert"],
    ["Mathematics is a game played according to certain simple rules with meaningless marks on paper.", "-David Hilbert"],
    ["Without mathematics, there's nothing you can do. Everything around you is mathematics. Everything around you is numbers.", "-Shakuntala Devi"]
    ]
     


function randomQuotes() {
    var idx = Math.floor(Math.random() * quotes.length);
    var Quote= quotes[idx];
    var quote= Quote[0];
    var author= Quote[1];
    document.getElementById('quote').innerHTML= quote;
    document.getElementById('author').innerHTML= author;
}