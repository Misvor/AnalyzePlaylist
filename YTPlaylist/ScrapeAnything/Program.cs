// See https://aka.ms/new-console-template for more information

using System.Text;

var uriString ="https://www.youtube.com/c/Veorra/videos";
//var uriString = Console.ReadLine();

try
{

}
catch (Exception ex)
{
    Console.WriteLine("Incorrect uri provided");
    Console.WriteLine(ex.Message);
}

Console.WriteLine("End!");

