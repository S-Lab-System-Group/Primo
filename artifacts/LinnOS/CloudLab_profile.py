"""Start a experiment for Primo-LinnOS Artifact.

Instructions:
Wait for the experiment to start, and then log into one or more of the nodes
by clicking on them in the toplogy, and choosing the `shell` menu option.
Use `sudo` to run root commands. 
"""

import geni.portal as portal
import geni.rspec.pg as pg
import geni.rspec.emulab as emulab

pc = portal.Context()
request = pc.makeRequestRSpec()

node = request.RawPC("node")
node.component_id = "urn:publicid:IDN+utah.cloudlab.us+node+pcap1"
node.Desire("pcap1", "1.0")

pc.printRequestRSpec(request)
