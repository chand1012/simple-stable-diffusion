const spinner = `<div id="loading-spinner" class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>`;

// Function to be executed once page is fully loaded
async function onFullyLoaded() {
  const resp = await fetch("/model_list");
  const data = await resp.json();
  // populate the model select
  const modelSelect = document.getElementById("model-select");
  // key of the model is the option value, value of the model is the option text
  for (const [key, value] of Object.entries(data)) {
    const option = document.createElement("option");
    option.value = key;
    option.text = value;
    modelSelect.appendChild(option);
  }
  // set the current value to 'dreamlike'
  modelSelect.value = "dreamlike";
  // remove disabled attribute
  modelSelect.removeAttribute("disabled");
}

// Add event listener for 'load' event to the window
window.addEventListener("load", onFullyLoaded);

// create a function to run on form submit
async function onSubmit(event) {
  // prevent the default form submission
  event.preventDefault();
  // append a spinner to the images row
  const imagesRow = document.getElementById("images");
  imagesRow.innerHTML += spinner;
  // disable the submit button
  const submitButton = document.getElementById("submit-button");
  submitButton.setAttribute("disabled", true);
  // get the form data
  const formData = new FormData(event.target);
  // get the form action
  const formAction = event.target.action;
  // convert so it fits the format of the api
  /*
            {
                "prompt": "string",
                "negative_prompt": "",
                "add_trigger": true,
                "opts": {
                    "guidance_scale": 7.5,
                    "height": 512,
                    "num_inference_steps": 50,
                    "width": 512
                }
            }
            */
  const data = {
    prompt: formData.get("prompt"),
    negative_prompt: formData.get("negative-prompt"),
    add_trigger: true,
    opts: {
      guidance_scale: parseFloat(formData.get("guidance-scale")),
      height: parseInt(formData.get("height")),
      num_inference_steps: parseInt(formData.get("steps")),
      width: parseInt(formData.get("width")),
    },
  };
  console.log(data);
  const model = formData.get("model-select");
  // post it to /generate_image/
  const resp = await fetch(`/generate_image/?model=${model}&response_format=b64_json`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  // get the response as json
  const json = await resp.json();
  // response is in format {"b64_json": "base64 encoded jpg"}
  // add a new image to the page with the base64 encoded jpeg to the images row
  // Create a new column element
  const col = document.createElement("div");
  col.classList.add("col-sm-2");

  // Create a new image element
  const img = document.createElement("img");
  img.src = `data:image/jpeg;base64,${json.b64_json}`;
  img.alt = "Generated Image";
  img.classList.add("img-fluid");

  // Create a new anchor element
  const anchor = document.createElement("a");
  anchor.href = img.src;
  anchor.target = "_blank";

  // Append the image to the anchor, and the anchor to the column
  anchor.appendChild(img);
  col.appendChild(anchor);

  // Assume `imagesRow` is the row to which you're adding the column
  imagesRow.appendChild(col);

  // remove all spinners
  const spinners = document.querySelectorAll("#loading-spinner");
  spinners.forEach((spinner) => spinner.remove());
  // enable the submit button
  submitButton.removeAttribute("disabled");
}
document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("main-form");
  form.addEventListener("submit", onSubmit);
});
