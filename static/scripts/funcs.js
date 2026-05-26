async function askAI(prompt) {
	const url = "http://127.0.0.1:8080/prompt";
	try {
		const response = await fetch(url, {
			method: "POST",
			headers: {
				"Content-Type": "application/json"
			},
			body: JSON.stringify({
				prompt: prompt
			})
		});

		if (!response.ok) {
			throw new Error(`Response status: ${response.status}`);
		}

		const json = await response.json();
		const promptContainer = document.querySelector('.promptContainer');
		const promptDiv = document.createElement('div');
		promptDiv.className = 'prompt';
		promptDiv.innerHTML = `
			<div class="promptHeader">Model: ${document.getElementById("model").value}</div>
			<div class="promptContent">${prompt}</div>
		`;

		const responseDiv = document.createElement('div');
		responseDiv.className = 'modelResponse';
		responseDiv.innerHTML = `<div class="modelResponseContent">${
			json.response
				.replace(/<think>([\s\S]*?)<\/think>/g, '<i class="modelThought">Thought... $1</i>')
				.replace(/\n/g, '<br>')
				.replace(/\t/g, '&nbsp;&nbsp;&nbsp;&nbsp;')
		}</div>`;

		promptContainer.appendChild(promptDiv);
		promptContainer.appendChild(responseDiv);
		document.getElementById('promptInput').removeAttribute('readonly');
		document.getElementById('promptInput').placeholder = `Ask anything...`;
		document.getElementById('promptInput').value = '';
		document.getElementById('promptInput').focus();
		promptContainer.scrollTop = promptContainer.scrollHeight;
		// console.log(json.response);
	} catch (error) {
		console.error(error.message);
		// Consider showing error to user in UI
	}
}

document.addEventListener("DOMContentLoaded", (evt) => {
	document.getElementById('promptInput').value = '';
	document.getElementById('promptInput').focus()
}, true);

document.getElementById('promptInput').addEventListener('keypress', (evt) => {
	if (evt.key === 'Enter') {
		askAI(document.getElementById("promptInput").value);
		document.getElementById('promptInput').setAttribute('readonly', true);
		document.getElementById('promptInput').value = `Asking ${document.getElementById("model").value}...`;
	}
});

document.getElementById("model").addEventListener("change", (evt) => {
	document.getElementById("promptInput").focus();
});