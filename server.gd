extends Node

@export var websocket_url = "ws://localhost:8765"
var socket = WebSocketPeer.new()

func _ready():
	var err = socket.connect_to_url(websocket_url)
	if err != OK:
		print("Unable to connect")
		set_process(false)
	else:
		await get_tree().create_timer(2).timeout
		socket.send_text("Client connected")

func _process(_delta):
	socket.poll()
	var state = socket.get_ready_state()
	# Tutaj dodastajemy przesyłkę w postaci informacji jaki sygnał osbłużyć:		
	if state == WebSocketPeer.STATE_OPEN:
		while socket.get_available_packet_count():
			var msg = get_msg()
			print(msg)
	elif state == WebSocketPeer.STATE_CLOSING:
		print("Websocket is closing")
	elif state == WebSocketPeer.STATE_CLOSED:
		var code = socket.get_close_code()
		print("WebSocket closed with code: %d. Clean: %s" % [code, code != -1])
		set_process(false) 
	
func send_msg(msg) ->void:
	socket.send_text(msg)

func get_msg() -> String:
	var msg = socket.get_packet().get_string_from_utf8()
	return msg
