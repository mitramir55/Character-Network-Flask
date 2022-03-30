
// for the sidebar
$(function() {
  // Sidebar toggle behavior
  $('#sidebarCollapse').on('click', function() {
    $('#sidebar, #content').toggleClass('active');
  });
});

var coll = document.getElementsByClassName("collapsible");
var i;

for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var content_ = this.nextElementSibling;
    if (content_.style.display === "block") {
      content_.style.display = "none";
    } else {
      content_.style.display = "block";
    }
  });
}

console.log(d3);
