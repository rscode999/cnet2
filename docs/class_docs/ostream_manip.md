# Output Stream Manipulators

[Back to central documentation](../documentation.md)

Special functions that change the behavior of output streams.

---
---
---

#### verbose

*Signature:* `template<typename CharT, typename Traits> inline std::basic_ostream<CharT, Traits>& verbose(std::basic_ostream<CharT, Traits>& output_stream)`

Sets the given output stream to print more information about a network and its components.

Usage example:
```
using namespace std;
using namespace cast;

Network net;

Add components to the network...

cout << net << endl; //Prints basic information about the network

cout << verbose; //Sets `cout` to print more

cout << net << endl; //Prints more detailed information
```

---

#### noverbose

*Signature:* `template<typename CharT, typename Traits> inline std::basic_ostream<CharT, Traits>& noverbose(std::basic_ostream<CharT, Traits>& output_stream)`

Stops an output stream from printing detailed network information.

Usage example:
```
using namespace std;
using namespace cast;

cout << verbose; //Sets `cout` to print more information

cout << noverbose; //Resets `cout` to its default
```