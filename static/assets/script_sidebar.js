

var bars = document.querySelectorAll("[barname]");
var path = window.location.pathname;
var page_name = path.split("/").pop();

for (i = 0; i < bars.length; i++) {
  var bar_name = bars[i].getAttribute("barname");
  if (bar_name==page_name) {
  bars[i].style.backgroundColor ="#DCDCDC";
  }
}
