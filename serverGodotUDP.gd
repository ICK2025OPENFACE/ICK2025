extends Node

var server = UDPServer.new()
var peers = []
var control_vals = []

signal client_control(control_vals)

func _ready():
	# Initialize control values
	control_vals.resize(4)
	control_vals.fill(0.0)
	
	# Start listening on port 4242
	server.listen(4242)
	print("Server started on port 4242")

func _process(delta):
	# Poll the server to handle connections and packets
	server.poll()

	# Handle new connections
	if server.is_connection_available():
		var peer = server.take_connection()
		print("New connection: %s:%s" % [peer.get_packet_ip(), peer.get_packet_port()])
		peers.append(peer)

	# Handle packets from connected peers
	for peer in peers:
		while true:
			var packet = peer.get_packet()
			if packet.is_empty():
				break  # No more packets for this peer
			
			# Extract packet data
			var id = (packet.slice(0, 1)).get_string_from_utf8()
			var packet_string = (packet.slice(1)).get_string_from_utf8()
			print("Received data from %s:%s -> ID: %s, Value: %s" % [
				peer.get_packet_ip(), peer.get_packet_port(), id, packet_string])
			
			# Update control values and emit signal
			control_vals[int(id)] = packet_string.to_float()
			emit_signal("client_control", control_vals)
