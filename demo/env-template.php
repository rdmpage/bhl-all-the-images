<?php

// For local testing put values here, rename this file to env.php, and keep
// env.php out of git (see demo/.gitignore).
// For production (e.g. Heroku) set these as environment variables instead.

putenv('BHL_SEARCH_API=http://YOUR-HETZNER-IP:8000');  // the search API base URL
putenv('BHL_SEARCH_KEY=');                             // must match the server's
                                                       // BHL_SEARCH_KEY (sent as
                                                       // the X-API-Key header)

?>
