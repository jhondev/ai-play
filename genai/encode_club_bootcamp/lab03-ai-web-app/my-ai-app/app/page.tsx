"use client";

import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function Chat() {
  const options = [
    { text: "Text", href: "/text" },
    { text: "Image", href: "/image" },
    { text: "Voice", href: "/voice" },
    { text: "Video", href: "/video" },
  ];
  return (
    <div className="grid grid-cols-3">
      {options.map((option) => (
        <Button variant={"outline"} key={option.href} asChild>
          <Link href={option.href}>{option.text}</Link>
        </Button>
      ))}
    </div>
  );
}
