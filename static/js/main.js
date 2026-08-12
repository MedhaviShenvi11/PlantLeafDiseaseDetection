// Image Preview

const imageInput=document.getElementById("imageInput");

if(imageInput){

imageInput.onchange=function(e){

const preview=document.getElementById("preview");

preview.src=URL.createObjectURL(e.target.files[0]);

preview.style.display="block";

}

}